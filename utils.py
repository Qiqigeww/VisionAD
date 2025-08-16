import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from dataset import MVTecDataset
from torch.nn import functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve, \
    average_precision_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter, binary_dilation
import os
from functools import partial
import math

import pickle
import torchvision.transforms.functional as TF

class BCE(torch.nn.Module):
    def __init__(self, alpha=1, beta=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        target = target.float()
        alpha = torch.tensor(self.alpha, device=pred.device)
        beta = torch.tensor(self.beta, device=pred.device)
        weight = torch.where(torch.eq(target, 1.), alpha, beta)
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        loss = (bce * weight).mean()
        return loss


class FocalBCE(torch.nn.Module):
    def __init__(self, gamma=2., alpha=1, beta=1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        target = target.float()
        alpha = torch.tensor(self.alpha, device=pred.device)
        beta = torch.tensor(self.beta, device=pred.device)

        alpha_factor = torch.where(target == 1., alpha, beta)
        focal_weight = torch.where(target == 1., 1. - pred, pred)
        focal_weight = torch.pow(focal_weight, self.gamma) * alpha_factor
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        loss = (bce * focal_weight).mean()
        return loss


class DiceLoss(torch.nn.Module):
    def __init__(self, eps=1., alpha=0.5, beta=0.5):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pred = pred.flatten(start_dim=1)
        target = target.flatten(start_dim=1)

        tp = torch.sum(pred * target, dim=1)
        fp = torch.sum(pred * (1. - target), dim=1)
        fn = torch.sum((1 - pred) * target, dim=1)
        dice_score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        dice_loss = 1 - dice_score
        return dice_loss.mean()


class DiceLossBatch(torch.nn.Module):
    def __init__(self, eps=1., alpha=0.5, beta=0.5):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        tp = torch.sum(pred * target)
        fp = torch.sum(pred * (1. - target))
        fn = torch.sum((1 - pred) * target)
        dice_score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        dice_loss = 1 - dice_score
        return dice_loss.mean()


def distance_loss(distance, gt):
    distance = distance.view(-1)
    gt = gt.view(-1)
    distance = distance[gt == 0]
    loss = distance.mean()
    return loss


def global_cosine(a, b, stop_grad=True):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        if stop_grad:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1).detach(),
                                            b[item].view(b[item].shape[0], -1)))
        else:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))
    loss = loss / len(a)
    return loss


def region_cosine(a, b, stop_grad=True):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        if stop_grad:
            loss += torch.mean(1 - cos_loss(a[item].detach(), b[item]))
        else:
            loss += torch.mean(1 - cos_loss(a[item], b[item]))
    loss = loss / len(a)
    return loss


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N


from sklearn.metrics import pairwise
def support_filter(dataloader, support_data_all, model, device):
    # 获取 dataloader 里 batch 的第一个样本（即 batch[0][0]）
    random_img_class = next(iter(dataloader))[0][0].unsqueeze(0)  # 只取一个样本，并保持维度
    
    # 只选取 support_data_all 每个类别的第一个样本
    support_keys = list(support_data_all.keys())  # 记录类别 key
    support_data_all_selected = torch.cat([support_data_all[key][0].unsqueeze(0) for key in support_keys], dim=0)

    # 计算 CLS token
    with torch.no_grad():
        support_cls_token = model.encoder(support_data_all_selected.to(device))  # 支持样本的特征
        query_cls_token = model.encoder(random_img_class.to(device))  # 只计算一个查询样本的特征

    # 计算余弦距离
    cos = pairwise.cosine_distances(support_cls_token.cpu(), query_cls_token.cpu())
    cos = cos.squeeze()  # 去掉多余的维度，使其变成 1D 数组
    
    # 获取最小值及其索引（最相似的类别）
    index = np.argmin(cos)  # 直接找最小值索引

    # 找到最匹配的类别，并返回对应的 support 数据
    matched_key = support_keys[index]  # 取出类别 key
    matched_data = support_data_all[matched_key]  # 只返回该类别的第一个样本
    
    return matched_data.unsqueeze(dim=0).to(device)


def evaluation_batch_1(model, dataloader, support_data_all, device, max_ratio=0, resize_mask=None):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []

    # supports, _ = next(iter(supportloader))
    # supports = supports.unsqueeze(dim=0).to(device)
    supports = support_filter(dataloader, support_data_all, model, device)

    memorize_supports = True
    use_memory = False
    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            # en, de = model(img, supports, memorize_supports, use_memory)
            # _, anomaly_map = cal_anomaly_maps(en, de, img.shape[-1])

            anomaly_map_ori = model(img, supports, memorize_supports, use_memory)
           
            # # 变换色彩域之后的预测结果
            img_aug1 = color_jitter_batch(img, 0.9)
            supports_aug1 =color_jitter_batch(supports, 0.9)
            anomaly_map_aug1 = model(img_aug1, supports_aug1, memorize_supports, use_memory, aug=1)
            anomaly_map_aug1 = torch.flip(anomaly_map_aug1, dims=[-2])  # 沿着高度翻转回来 
            
            img_aug2 = value_limit(img)
            supports_aug2 =value_limit(supports)
            anomaly_map_aug2 = model(img_aug2, supports_aug2, memorize_supports, use_memory, aug=2)            
               
            # # # 将2者预测相加
            # anomaly_map = 0.7 * anomaly_map_ori + 0.3 * anomaly_map_aug2
            # anomaly_map = 0.6 * anomaly_map_ori + 0.2 * anomaly_map_aug1 + 0.2 * anomaly_map_aug2
            anomaly_map = anomaly_map_ori + anomaly_map_aug1 + anomaly_map_aug2
            # anomaly_map = anomaly_map_ori
                      
            memorize_supports = False
            use_memory = True

            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')

            gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        aupro_px = compute_pro(gt_list_px, pr_list_px)

        gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()

        auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        ap_px = average_precision_score(gt_list_px, pr_list_px)
        ap_sp = average_precision_score(gt_list_sp, pr_list_sp)

        f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def evaluation_batch(model, dataloader, supportloader, device, max_ratio=0, resize_mask=None):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []

    supports, _ = next(iter(supportloader))
    supports = supports.unsqueeze(dim=0).to(device)

    memorize_supports = True
    use_memory = False
    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            # en, de = model(img, supports, memorize_supports, use_memory)
            # _, anomaly_map = cal_anomaly_maps(en, de, img.shape[-1])

            anomaly_map_ori = model(img, supports, memorize_supports, use_memory)
           
            # # 变换色彩域之后的预测结果
            img_aug1 = color_jitter_batch(img, 0.9)
            supports_aug1 =color_jitter_batch(supports, 0.9)
            anomaly_map_aug1 = model(img_aug1, supports_aug1, memorize_supports, use_memory, aug=1)
            anomaly_map_aug1 = torch.flip(anomaly_map_aug1, dims=[-2])  # 沿着高度翻转回来 
            
            img_aug2 = value_limit(img)
            supports_aug2 =value_limit(supports)
            anomaly_map_aug2 = model(img_aug2, supports_aug2, memorize_supports, use_memory, aug=2)            
               
            # # # 将2者预测相加
            # anomaly_map = 0.7 * anomaly_map_ori + 0.3 * anomaly_map_aug2
            # anomaly_map = 0.6 * anomaly_map_ori + 0.2 * anomaly_map_aug1 + 0.2 * anomaly_map_aug2
            anomaly_map = anomaly_map_ori + anomaly_map_aug1 + anomaly_map_aug2
            # anomaly_map = anomaly_map_ori
                      
            memorize_supports = False
            use_memory = True

            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')

            gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        aupro_px = compute_pro(gt_list_px, pr_list_px)

        gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()

        auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        ap_px = average_precision_score(gt_list_px, pr_list_px)
        ap_sp = average_precision_score(gt_list_sp, pr_list_sp)

        f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]

def evaluation_batch_vis(model, dataloader, supportloader, device, max_ratio=0, resize_mask=None):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    img_path_list = []

    supports, _ = next(iter(supportloader))
    supports = supports.unsqueeze(dim=0).to(device)

    memorize_supports = True
    use_memory = False
    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            # en, de = model(img, supports, memorize_supports, use_memory)
            # _, anomaly_map = cal_anomaly_maps(en, de, img.shape[-1])

            anomaly_map_ori = model(img, supports, memorize_supports, use_memory)
           
            # # 变换色彩域之后的预测结果
            img_aug1 = color_jitter_batch(img, 0.9)
            supports_aug1 =color_jitter_batch(supports, 0.9)
            anomaly_map_aug1 = model(img_aug1, supports_aug1, memorize_supports, use_memory, aug=1)
            anomaly_map_aug1 = torch.flip(anomaly_map_aug1, dims=[-2])  # 沿着高度翻转回来 
            
            img_aug2 = value_limit(img)
            supports_aug2 =value_limit(supports)
            anomaly_map_aug2 = model(img_aug2, supports_aug2, memorize_supports, use_memory, aug=2)            
               
            # # # 将2者预测相加
            # anomaly_map = 0.7 * anomaly_map_ori + 0.3 * anomaly_map_aug2
            # anomaly_map = 0.6 * anomaly_map_ori + 0.2 * anomaly_map_aug1 + 0.2 * anomaly_map_aug2
            anomaly_map = anomaly_map_ori + anomaly_map_aug1 + anomaly_map_aug2
            # anomaly_map = anomaly_map_ori
                      
            memorize_supports = False
            use_memory = True

            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')

            gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)
            
            img_path_list.extend(img_path)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        aupro_px = 0    #   compute_pro(gt_list_px, pr_list_px)

        #-------------------------------------------------------------------------------------------  
        from test_help import visualize_compound_1        
        visualize_compound_1(
                img_path_list,
                pr_list_px, gt_list_px,
                f"./vis_compound_1/{img_path_list[0].split('/')[-5]}"
            )
        print('We have saved the predicted images.\n \n \n')          
        #------------------------------------------------------------------------------------------- 

        gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()

        auroc_px = 0    #   roc_auc_score(gt_list_px, pr_list_px)
        auroc_sp = 0    #   roc_auc_score(gt_list_sp, pr_list_sp)
        ap_px = 0   #  average_precision_score(gt_list_px, pr_list_px)
        ap_sp = 0   #  average_precision_score(gt_list_sp, pr_list_sp)

        f1_sp = 0   #  f1_score_max(gt_list_sp, pr_list_sp)
        f1_px = 0   #  f1_score_max(gt_list_px, pr_list_px)       

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]

# 对批量进行处理
def color_jitter_batch(augmented_image, value=0.9):
    """
    对包含额外维度的 Tensor 图像批量应用固定参数的 ColorJitter。
    :param tensor_image_batch: 输入的 Tensor 图像批量，形状为 [B, N, C, H, W]
    :param brightness: 亮度调整系数
    :param contrast: 对比度调整系数
    :param saturation: 饱和度调整系数
    :return: 调整后的批量 Tensor 图像
    """
    brightness = value
    contrast = value
    saturation = value
    sharpness_factor = value
    gamma = value
    # augmented_image = TF.adjust_brightness(augmented_image, brightness)
    # augmented_image = TF.adjust_contrast(augmented_image, contrast)
    # augmented_image = TF.adjust_saturation(augmented_image, saturation)
    # augmented_image = TF.adjust_sharpness(augmented_image, sharpness_factor)
    # augmented_image = TF.adjust_gamma(augmented_image, gamma)
    augmented_image = torch.flip(augmented_image, dims=[-2])
    # augmented_image = augmented_image[..., [2, 1, 0], :, :]  # 通道交换
    return augmented_image

def value_limit(value):
    
    value = value.clamp(min=0)
    return value
    

def count_Params_macs_I(model, dataloader, supportloader, device, max_ratio=0, resize_mask=None):
    model.eval()
    
    from thop import profile
    from thop import clever_format
    import time
    def get_timepc():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    supports, _ = next(iter(supportloader))
    supports = supports.unsqueeze(dim=0).to(device)

    memorize_supports = True
    use_memory = False
    i = 0
    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)


            anomaly_map_ori = model(img, supports, memorize_supports, use_memory)
           
            # # 变换色彩域之后的预测结果
            img_aug1 = color_jitter_batch(img, 0.9)
            supports_aug1 =color_jitter_batch(supports, 0.9)
            anomaly_map_aug1 = model(img_aug1, supports_aug1, memorize_supports, use_memory, aug=1)
            anomaly_map_aug1 = torch.flip(anomaly_map_aug1, dims=[-2])  # 沿着高度翻转回来 
            
            img_aug2 = value_limit(img)
            supports_aug2 =value_limit(supports)
            anomaly_map_aug2 = model(img_aug2, supports_aug2, memorize_supports, use_memory, aug=2)            
               
            # # # 将2者预测相加
            anomaly_map = anomaly_map_ori + anomaly_map_aug1 + anomaly_map_aug2
                      
            memorize_supports = False
            use_memory = True

            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')

            gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)
            i += 1
            if i == 6:
                t_s = get_timepc()
            if i == 16:
                t_e = get_timepc()
                macs, params = profile(model, inputs=(img, supports, memorize_supports, use_memory))
                macs, params = clever_format([macs/img.shape[0], params], "%.1f")
                break
    # return (2*flops)/1e9, params/1e6, img.shape[0] * 10 / (t_e - t_s)
    return macs, params, img.shape[0] * 10 / (t_e - t_s)

def visualize(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)

            for i in range(0, anomaly_map.shape[0], 8):
                heatmap = min_max_norm(anomaly_map[i, 0].cpu().numpy())
                heatmap = cvt2heatmap(heatmap * 255)
                im = img[i].permute(1, 2, 0).cpu().numpy()
                im = im * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                im = (im * 255).astype('uint8')
                im = im[:, :, ::-1]
                hm_on_img = show_cam_on_image(im, heatmap)
                mask = (gt[i][0].numpy() * 255).astype('uint8')
                save_dir_class = os.path.join(save_dir, str(_class_))
                if not os.path.exists(save_dir_class):
                    os.mkdir(save_dir_class)
                name = img_path[i].split('/')[-2] + '_' + img_path[i].split('/')[-1].replace('.png', '')
                cv2.imwrite(save_dir_class + '/' + name + '_img.png', im)
                cv2.imwrite(save_dir_class + '/' + name + '_cam.png', hm_on_img)
                cv2.imwrite(save_dir_class + '/' + name + '_gt.png', mask)

    return


def save_feature(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./feature', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)

            en_abnorm_list = []
            en_normal_list = []
            de_abnorm_list = []
            de_normal_list = []

            for i in range(3):
                en_feat = en[0 + i]
                de_feat = de[0 + i]

                gt_resize = F.interpolate(gt, size=en_feat.shape[2], mode='bilinear') > 0

                en_abnorm = en_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0]]
                en_normal = en_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0] == 0]

                de_abnorm = de_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0]]
                de_normal = de_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0] == 0]

                en_abnorm_list.append(F.normalize(en_abnorm, dim=1).cpu().numpy())
                en_normal_list.append(F.normalize(en_normal, dim=1).cpu().numpy())
                de_abnorm_list.append(F.normalize(de_abnorm, dim=1).cpu().numpy())
                de_normal_list.append(F.normalize(de_normal, dim=1).cpu().numpy())

            save_dir_class = os.path.join(save_dir, str(_class_))
            if not os.path.exists(save_dir_class):
                os.mkdir(save_dir_class)
            name = img_path[0].split('/')[-2] + '_' + img_path[0].split('/')[-1].replace('.png', '')

            saved_dict = {'en_abnorm_list': en_abnorm_list, 'en_normal_list': en_normal_list,
                          'de_abnorm_list': de_abnorm_list, 'de_normal_list': de_normal_list}

            with open(save_dir_class + '/' + name + '.pkl', 'wb') as f:
                pickle.dump(saved_dict, f)

    return


def visualize_noseg(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with torch.no_grad():
        for img, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            heatmap = min_max_norm(anomaly_map)
            heatmap = cvt2heatmap(heatmap * 255)
            img = img.permute(0, 2, 3, 1).cpu().numpy()[0]
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = (img * 255).astype('uint8')
            hm_on_img = show_cam_on_image(img, heatmap)

            save_dir_class = os.path.join(save_dir, str(_class_))
            if not os.path.exists(save_dir_class):
                os.mkdir(save_dir_class)
            name = img_path[0].split('/')[-2] + '_' + img_path[0].split('/')[-1].replace('.png', '')
            cv2.imwrite(save_dir_class + '/' + name + '_seg.png', heatmap)
            cv2.imwrite(save_dir_class + '/' + name + '_cam.png', hm_on_img)

    return


def visualize_loco(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    with torch.no_grad():
        for img, gt, label, img_path, defect_type, size in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            anomaly_map = cv2.resize(anomaly_map, dsize=(size[0].item(), size[1].item()),
                                     interpolation=cv2.INTER_NEAREST)

            save_dir_class = os.path.join(save_dir, str(_class_), 'test', defect_type[0])
            if not os.path.exists(save_dir_class):
                os.makedirs(save_dir_class)
            name = img_path[0].split('/')[-1].replace('.png', '')
            cv2.imwrite(save_dir_class + '/' + name + '.tiff', anomaly_map)
    return


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class FeatureJitter(torch.nn.Module):
    def __init__(self, scale=1., p=0.25) -> None:
        super(FeatureJitter, self).__init__()
        self.scale = scale
        self.p = p

    def add_jitter(self, feature):
        if self.scale > 0:
            B, C, H, W = feature.shape
            feature_norms = feature.norm(dim=1).unsqueeze(1) / C  # B*1*H*W
            jitter = torch.randn((B, C, H, W), device=feature.device)
            jitter = F.normalize(jitter, dim=1)
            jitter = jitter * feature_norms * self.scale
            mask = torch.rand((B, 1, H, W), device=feature.device) < self.p
            feature = feature + jitter * mask
        return feature

    def forward(self, x):
        if self.training:
            x = self.add_jitter(x)
        return x


def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new)


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
