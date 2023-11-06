import logging
import os
import sys
import math
import tempfile
from glob import glob
import matplotlib.pyplot as plt
import torch
from PIL import Image

import monai
from monai.networks.nets.basic_unetplusplus import BasicUNetPlusPlus
from monai.data import list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)

def plot_loss_and_metric(epoch_num, epoch_loss_values, metric_values):
    fig = plt.figure("train",(12, 6))
    # 绘制子图
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = list(range(1, epoch_num+1))
    y1 = epoch_loss_values
    plot_data_1 = [x, y1]
    plt.xlabel("Epoch")
    plt.plot(plot_data_1[0],plot_data_1[1])
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    y2 = metric_values
    plot_data_2 = [x, y2]
    plt.xlabel("Epoch")
    plt.plot(plot_data_2[0],plot_data_2[1], color='red')
    # 保存
    fig.savefig('loss_and_metric.png')
    

def main(img_file, mask_file, epoch_num=10):
    '''
    img_file:所有图片文件的根目录
    mask_file:所有掩膜文件的根目录
    '''
    # 打印配置和日志信息
    # monai.config.print_config()
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # 图片和掩膜文件路径
    images = sorted(glob(os.path.join(img_file, "*.png")))
    segs = sorted(glob(os.path.join(mask_file, "*.png")))
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:100], segs[:100])]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images[100:120], segs[100:120])]
    
    # 数据增强
    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]),
            # keys: 这是一个列表，指定了要应用此数据转换的数据字段，通常包括图像和对应的分割掩模。
            # label_key: 指定了用于识别正样本和负样本的标签的字段。
            # spatial_size: 指定了随机裁剪出的区域的空间尺寸（高度和宽度）。
            # pos: 指定了每次裁剪后的图片中要保留的正样本的数量。
            # neg: 指定了每次裁剪后的图片中要保留的负样本的数量。
            # num_samples: 指定了每个输入图像应生成的样本数量。
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=[512, 512], 
                pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]),
        ]
    )

    # 定义dataset和dataloader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=3,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    # val_loader的batch_size参数只能为1，因为这里传入的是原始图片，尺寸不一样，不能stack
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=14, collate_fn=list_data_collate)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(softmax=True), AsDiscrete(threshold=0.5)])

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicUNetPlusPlus(
        spatial_dims=2,
        in_channels=3,
        out_channels=22,
        features=(16, 32, 64, 128, 256, 16),
    ).to(device)
    loss_function = monai.losses.DiceLoss(softmax=True, to_onehot_y=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    ###################正式开始训练###################
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    train_metric_values = list()
    metric_values = list()
    for epoch in range(epoch_num):
        print("-" * epoch_num)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        ######### 训练 ##########
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = math.ceil(len(train_ds) / train_loader.batch_size)  # 每个epoch中有多少个batch
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step  # 每个epoch的平均损失
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        ######### 验证 ##########
        model.eval()
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None
            for val_data in val_loader:
                val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                roi_size = (256, 256)  # 滑动窗口推断时输入图像的区域大小。
                sw_batch_size = 4  # sliding_window_batch_size，一次性输入到模型的小块数量
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]  # decollate_batch将模型输出中的这些小块分开，以便后续分析或可视化。
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()
            metric_values.append(metric)
            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                # 获取当前py文件的文件名
                current_file_path = __file__          
                file_name = os.path.basename(current_file_path)
                torch.save(model.state_dict(), f"best_model_of_{file_name}.pth")
                print("saved new best metric model")
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )
    plot_loss_and_metric(epoch_num, epoch_loss_values, metric_values)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    

if __name__ == "__main__":
    img_file = '/root/autodl-tmp/Dataset/CrowdsourcingDataset-Amgadetal2019/images'
    mask_file = '/root/autodl-tmp/Dataset/CrowdsourcingDataset-Amgadetal2019/masks'
    torch.multiprocessing.set_start_method('spawn')
    main(img_file, mask_file, epoch_num=10)
