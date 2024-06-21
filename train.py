# 导入相应的库

import json
import math
import os
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from tqdm import tqdm

from dataLoader.dataSet import read_split_data
from dataLoader.dataLoader import My_Dataset
from utils import train_one_epoch,  get_params_groups, create_lr_scheduler, save_checkpoint, \
    save_metrics, EMA, evaluate_with_tta,seed_everything
from models.FHBNet import create_FHBNet
data_characteristics = {
    'all_raters': {
        'height_mean': 2731.472954699121,
        'width_mean': 1003.7423935091277,
        'ch_mean':[0.44865146, 0.48655152, 0.37425962],
        'ch_std':[0.24161708, 0.23912759, 0.21776654]
    }
}

# 主函数
def main(opt):
    # 1.读取一些配置参数，并且输出
    print(opt)
    assert os.path.exists(opt.data_path), "{} dose not exists.".format(opt.data_path)

    # 创建日志文件
    tb_writer = SummaryWriter()
    # 日志保存路径
    save_dir = tb_writer.log_dir
    # save_dir="/root/autodl-tmp/"+tb_writer.log_dir
    print(save_dir)
    # 模型保存路径
    weights_dir = save_dir + "/weights"
    # 如果文件夹不存在，则创建文件夹
    os.makedirs(weights_dir, exist_ok=True)

    # Save command-line arguments to a file
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as config_file:
        json.dump(vars(opt), config_file, indent=4)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 设备
    device = torch.device('cuda:0' if torch.cuda.is_available() and opt.use_cuda else "cpu")
    print(device)

    # 2.数据读取
    train_images_path, val_images_path, train_labels, val_labels, every_class_num = read_split_data(
        data_root=opt.data_path, val_rate=0.2, save_dir=save_dir)

    # 数据处理
    characteristics = data_characteristics['all_raters']
    means, stds = characteristics['ch_mean'], characteristics['ch_std']

    # 数据加载：dataset,transform,dataloader
    data_transform = {
        "train": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=(224, 224)),
                transforms.RandomRotation(degrees=2.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                # transforms.ColorJitter(
                #     brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05
                # ),
                transforms.Normalize(mean=means, std=stds)
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=(224, 224)),
                # transforms.CenterCrop(224),
                transforms.Normalize(mean=means, std=stds)
            ]
        )
    }

    train_dataset = My_Dataset(images_path=train_images_path, images_class=train_labels,
                               transform=data_transform['train'])
    val_dataset = My_Dataset(images_path=val_images_path, images_class=val_labels, transform=data_transform['val'])

    nw=opt.num_worker
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                                               num_workers=nw, collate_fn=train_dataset.collate_fn, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=True,
                                             num_workers=nw, collate_fn=val_dataset.collate_fn, pin_memory=True)

    # 网络搭建：model
    classes=1
    model=create_FHBNet(num_classes=classes)
    if opt.weights != '':
        assert os.path.exists(opt.weights), "weights file: '{}' not exist.".format(opt.weights)
        print('已加载预训练权重')
        weights_dict = torch.load(opt.weights)
        # del_keys = ['head.weight', 'head.bias']
        # for k in del_keys:
        #     del weights_dict[k]
        #     print("已删除头部全连接的权重")
        model.load_state_dict(weights_dict, strict=False)

    if opt.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    model = model.to(device)

    # EMA
    ema = EMA(model, 0.999)
    ema.register()

    #   3.3 优化器，学习率，更新策略,损失函数
    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=5e-2)
    if opt.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(pg, lr=opt.lr, momentum=0.9, weight_decay=1e-4)
    elif opt.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(pg, lr=opt.lr, weight_decay=1e-4)
    elif opt.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(pg, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-2)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - opt.lrf) + opt.lrf  # cosine
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)

    scheduler = create_lr_scheduler(optimizer, len(train_loader), opt.epochs,
                                    warmup=True, warmup_epochs=5)
    # 损失函数
    loss_function=nn.L1Loss()
    # loss_function=nn.MSELoss()

    # 模型训练：train
    best_acc = -np.inf
    best_epoch = 0

    # TTA
    tta_transforms = [
        RandomHorizontalFlip(p=1),  # 水平翻转
        RandomVerticalFlip(p=1),  # 垂直翻转
        RandomRotation(degrees=90),  # 旋转90度
    ]
    for epoch in tqdm(range(opt.epochs)):
        # train
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, loss_function, epoch=epoch,
                                                scheduler=scheduler, ema=ema)

        #  eval
        val_loss, val_acc, val_f1, val_recall, val_pre,val_rmse,val_rmse_per,pre_dict,rec_dict,f1_dict,val_r2,r2_dict = evaluate_with_tta(model, val_loader, device,                                                  loss_function,epoch, ema=ema, tta_transform=tta_transforms)
        print("第{}轮训练后:".format(epoch + 1))
        print("LOSS:", val_loss)
        print("ACC:", val_acc)
        print("F1:", val_f1)
        print("Recall:", val_recall)
        print("Precession", val_pre)
        print("R2",val_r2)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate", "images", "val_f1", "val_recall",
                "val_precession"]
        save_metrics(epoch + 1, train_loss, train_acc, val_loss, val_acc, val_f1, val_recall, val_pre,val_rmse,val_rmse_per,pre_dict,rec_dict,f1_dict,val_r2,r2_dict, save_dir)
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        batch_images = next(iter(train_loader))[0]
        tb_writer.add_images(tags[5], batch_images, epoch)
        tb_writer.add_scalar(tags[6], val_f1, epoch)
        tb_writer.add_scalar(tags[7], val_recall, epoch)
        tb_writer.add_scalar(tags[8], val_pre, epoch)
        #   3.6 模型保存：save
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch + 1
            print("best epoch：", best_epoch)
            model_path = weights_dir + "/best_model.pth"
            torch.save(model.state_dict(), model_path)
            print("best epoch参数已保存")
            save_checkpoint(model, optimizer, epoch + 1, val_loss, val_acc, val_f1, val_recall, val_pre, is_best,
                            save_dir, opt)
    tb_writer.close()


# 程序入口
if __name__ == '__main__':
    seed_everything(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default=r"",
                        help='The data path')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-worker', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.001)

    parser.add_argument('--weights', type=str, default=r"", help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--optimizer', type=str, default='adam')  # sgd,adam,adamw

    args = parser.parse_args()

    main(opt=args)
