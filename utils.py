import json
import math
import os
import shutil
import sys

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np
import random
from ttach import Compose, HorizontalFlip, VerticalFlip, Rotate90
from sklearn.metrics import f1_score, recall_score, precision_score, r2_score
from sklearn.metrics import precision_recall_fscore_support


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def matplotlib_imshow(img, one_channel=False):
    fig = plt.figure()
    if one_channel:
        img = img.mean(dim=0)
        plt.imshow(img.numpy(), cmap="Greys")
        # plt.show()
        return img

    else:
        img = img.numpy().transpose(1, 2, 0)
        unnorm_img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        # unnorm_img = img * 255
        img = img.astype('uint8')
        unnorm_img = unnorm_img.astype('uint8')
        norm_image = torch.Tensor(img).permute(2, 0, 1)
        plt.imshow(unnorm_img)
        # plt.savefig("train_images.jpg")
        # plt.show()
        return norm_image, fig


def plot_data_loader_image(data_loader):
    plt.show()


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def calculate_rmse_per_class(predictions, targets, num_classes=3):
    # 计算每个类别的rmse
    rmse_per_class = []

    for class_label in range(num_classes):
        # 筛选出属于当前类别的预测值和真实标签
        class_predictions = predictions[targets == class_label]
        class_targets = targets[targets == class_label]

        # 计算当前类别的 RMSE
        rmse = calculate_rmse(class_predictions, class_targets)
        rmse_per_class.append(rmse)

    return rmse_per_class


def calculate_rmse(predictions, targets):
    """
    计算均方根误差（RMSE）

    参数:
    - predictions: 模型的预测值（张量）
    - targets: 实际目标值（张量）

    返回:
    - rmse: 均方根误差值
    """
    residuals = predictions - targets
    squared_residuals = residuals ** 2
    rmse = torch.sqrt(torch.mean(squared_residuals))
    return rmse.item()


# 数据后处理
def process_regression_output(predictions, targets):
    predictions = predictions.to('cpu')
    targets = targets.to('cpu')
    predictions = torch.where(
        predictions > targets.max(),
        torch.ones(size=predictions.size()) * targets.max(),
        predictions
    )
    predictions = torch.where(
        predictions < targets.min(),
        torch.ones(size=predictions.size()) * targets.min(),
        predictions
    )

    predictions = torch.round(predictions)
    return predictions


def calculate_metrics(true_labels, pred_classes, num_classes=6):
    # 计算 F1 Score、Precision 和 Recall
    f1 = f1_score(y_true=true_labels.cpu().numpy(), y_pred=pred_classes.cpu().numpy(), average='weighted')
    precision = precision_score(y_true=true_labels.cpu().numpy(), y_pred=pred_classes.cpu().numpy(), average='weighted')
    recall = recall_score(y_true=true_labels.cpu().numpy(), y_pred=pred_classes.cpu().numpy(), average='weighted',
                          zero_division=0)
    # 打印各个类别的Precision和Recall
    # for i in range(num_classes):
    #     print(f"Class {i}: Precision={precision[i]}, Recall={recall[i]}")
    return f1, precision, recall


def calculate_r2_per_class(all_targets, all_predictions, num_classes):
    r2_per_class = []

    for class_id in range(num_classes):
        # Select samples for the current class
        class_indices = (all_targets == class_id).nonzero(as_tuple=True)[0]

        if len(class_indices) >= 2:
            # At least two samples are needed for R² calculation
            class_targets = all_targets[class_indices]
            class_predictions = all_predictions[class_indices]

            # Calculate R² for the current class
            r2_class = r2_score(class_targets.cpu(), class_predictions.cpu())
            r2_per_class.append(r2_class)
        else:
            # Insufficient samples for R² calculation, append None
            r2_per_class.append(None)

    return r2_per_class


def train_one_epoch(model, data_loader, device, optimizer, loss_function, epoch, scheduler, ema):
    model.train()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    accu_predictions = []  # 用于存储所有预测值
    accu_targets = []  # 用于存储所有目标值
    sample_num = 0
    # data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = pred.squeeze(dim=1)
        loss = loss_function(pred_classes.to(device).float(), labels.to(device).float())
        pred_processed = process_regression_output(pred_classes, labels)
        accu_num += torch.eq(pred_processed.to(device), labels.to(device)).sum()
        loss.backward()
        accu_loss += loss.detach()  # accu_loss+=loss.item()
        accu_predictions.append(pred_processed.detach().cuda())  # 存储预测值
        accu_targets.append(labels.cuda())  # 存储目标值

        if step % 100 == 0:
            print(
                "train epoch {} step {} train loss: {:.5f} train acc: {:.5f} lr: {:.7f}".format(
                    epoch, step + 1, accu_loss.item() / (step + 1), accu_num.item() / sample_num,
                    optimizer.param_groups[0]["lr"]))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        ema.update()
        optimizer.zero_grad()
        scheduler.step()

    average_loss = accu_loss.item() / (step + 1)
    accuracy = accu_num.item() / sample_num

    return average_loss, accuracy

@torch.no_grad()
def evaluate_with_tta(model, data_loader, device, loss_function, epoch, ema, tta_transform=None):
    model.eval()
    print('----------开始验证-----------')
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    all_labels = []
    all_preds = []
    sample_num = 0
    normal_transformer = [
        transforms.Normalize(mean=[0.44865146, 0.48655152, 0.37425962],
                             std=[0.24161708, 0.23912759, 0.21776654])]
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        if tta_transform is not None:
            # Apply TTA transformations
            tta_images = []
            for transform in tta_transform:
                augmented_images = transform(images)
                for normal_t in normal_transformer:
                    augmented_images = normal_t(augmented_images)
                tta_images.append(augmented_images.to(device))
            for normal_t in normal_transformer:
                yuanlai_image = normal_t(images)
            tta_images.append(yuanlai_image)

            # Get predictions for each augmented image
            preds = [model(img.to(device)) for img in tta_images]

            # Average predictions
            pred = sum(preds) / len(preds)

        else:
            # No TTA, use original images
            pred = model(images.to(device))
        pred_classes = process_regression_output(pred, labels)
        pred_classes_1 = pred_classes
        pred_classes = pred_classes.squeeze(dim=1)
        accu_num += torch.eq(pred_classes.to(device), labels.to(device)).sum()
        loss = loss_function(pred_classes.to(device).float(), labels.to(device).float())
        accu_loss += loss
        # Collect labels and predictions for further metric calculation
        all_preds.append(pred_classes_1.detach().cuda())
        all_labels.append(labels.cuda())

    # 将所有预测值和目标值合并为一个张量
    all_predictions = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_labels, dim=0)

    # 计算并记录 RMSE
    rmse = calculate_rmse(all_predictions.squeeze(dim=1), all_targets)
    rmse_per = calculate_rmse_per_class(all_predictions.squeeze(dim=1), all_targets)

    average_loss = accu_loss.item() / (step + 1)
    accuracy = accu_num.item() / sample_num
    ema.apply_shadow()
    ema.restore()
    # 计算每个类别的 F1、Recall 和 Precision
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets.cpu(), all_predictions.cpu(), average=None)
    # 将结果转换为字典以便更容易访问
    precision_dict = {f'类别_{i}': precision[i] for i in range(len(precision))}
    recall_dict = {f'类别_{i}': recall[i] for i in range(len(recall))}
    f1_dict = {f'类别_{i}': f1[i] for i in range(len(f1))}
    f1, precision, recall = calculate_metrics(all_targets, all_predictions.squeeze().long())

    r2_value = r2_score(all_targets.cpu(), all_predictions.cpu())
    num_classes = 3
    # Calculate R² per class
    r2_per_class = calculate_r2_per_class(all_targets, all_predictions.squeeze().cpu(), num_classes)

    # Organize R² per class in a dictionary
    r2_dict_per_class = {f'类别_{i}': r2_per_class[i] for i in range(len(r2_per_class))}
    return average_loss, accuracy, f1, recall, precision, rmse, rmse_per, precision_dict, recall_dict, f1_dict, r2_value, r2_dict_per_class


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def save_checkpoint(model, optimizer, epoch, loss, acc, f1, recall, pre, is_best, save_dir, args):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'acc': acc,
        'F1': f1,
        'recall': recall,
        'precision': pre
    }

    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_model_path = os.path.join(save_dir, 'best_model.pth')
        shutil.copyfile(checkpoint_path, best_model_path)
        os.remove(checkpoint_path)
    # Save training configuration
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as config_file:
        json.dump(vars(args), config_file, indent=4)

def save_metrics(epoch, train_loss, train_acc, val_loss, val_acc, val_f1, val_recall, val_pre, val_rmse, val_rmse_per,
                 pre_dict,
                 rec_dict, f1_dict, val_r2, r2_dict, save_dir):
    metrics_path = os.path.join(save_dir, 'metrics.txt')
    metrics_path1 = os.path.join(save_dir, "metrics1.txt")
    mode = 'a' if os.path.exists(metrics_path) else 'w'
    mode1 = 'a' if os.path.exists(metrics_path1) else 'w'
    with open(metrics_path, mode) as file:
        if mode == 'w':
            file.write(
                "Epoch\tTrain Loss\tTrain Acc\tVal Loss\tVal Acc\tVal Recall\tVal F1\tVal Pre\tVal Rmse\tVal R2\n")

        file.write(
            f"{epoch}\t{train_loss:.6f}\t{train_acc:.4f}\t{val_loss:.6f}\t{val_acc:.4f}\t{val_recall:.4f}\t{val_f1:.4f}\t{val_pre:.4f}\t{val_rmse:.4f}\t{val_r2:.4f}\n")

    with open(metrics_path1, mode1) as file1:
        file1.write("Epoch：")
        file1.write(f"{epoch}\n")
        file1.write("Val Per_dict：\n")
        # If pre_dict is a list, write it directly
        file1.write("\t".join([f"{value}：{key}" for value, key in pre_dict.items()]) + "\t")

        file1.write("\n Val Recall_dict：\n")
        file1.write("\t".join([f"{value}：{key}" for value, key in rec_dict.items()]) + "\t")

        file1.write("\n Val F1_dict：\n")
        file1.write("\t".join([f"{value}：{key}" for value, key in f1_dict.items()]) + "\t")

        file1.write("\n Val rmse_dict：\n")
        file1.write("\t".join([f"{rmse}" for rmse in val_rmse_per]) + "\t")

        file1.write("\n Val r2_dict：\n")
        file1.write("\t".join([f"{value}：{key}" for value, key in r2_dict.items()]) + "\n")
