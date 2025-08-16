# This is a sample Python script.
# support的初始种子为1; 

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from dataset import get_data_transforms, get_alb_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset

from models.uad import DADE, DADEv2
from models import vit_encoder
from torch.nn.init import trunc_normal_
from models.vision_transformer import Block as VitBlock, Attention, CrossAttention, CrossBlock
from dataset import MVTecDataset, RealIADDatasetFADE
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, evaluation_batch_vis
from torch.nn import functional as F
from functools import partial
from optimizers import StableAdamW
import warnings
import copy
import logging
import itertools

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log_mvtec.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train():
    setup_seed(args.seed)

    total_iters = 100000
    batch_size = 8
    image_size = 448
    crop_size = 392

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    # train_transform = get_alb_transforms(image_size)

    """Test Data"""
    support_data_list = []
    test_data_list = []
    for i, item in enumerate(test_item_list):
        support_data = ImageFolder(root=os.path.join(args.test_data_path, item, 'train'), transform=data_transform)
        test_data = MVTecDataset(root=os.path.join(args.test_data_path, item), transform=data_transform,
                                 gt_transform=gt_transform, phase="test")
        support_data_list.append(support_data)
        test_data_list.append(test_data)

    # encoder_name = 'dinov2reg_vit_small_14'
    # encoder_name = 'dinov2reg_vit_base_14'
    encoder_name = 'dinov2reg_vit_large_14'
    # encoder_name = 'dinov2reg_vit_giant2_14'

    # encoder_name = 'dinov2_vit_base_14'
    # encoder_name = 'dino_vit_base_16'
    # encoder_name = 'ibot_vit_base_16'
    # encoder_name = 'mae_vit_base_16'
    # encoder_name = 'beitv2_vit_base_16'
    # encoder_name = 'beit_vit_base_16'
    # encoder_name = 'digpt_vit_base_16'
    # encoder_name = 'deit_vit_base_16'
    # encoder_name = 'mocov3_vit_base_16'
    # encoder_name = 'DINOV2_proteus_vit_base_14'
    # encoder_name = 'DINOV2_proteus_vit_large_14'
    # encoder_name = 'DINOV2_proteus_vit_small_14'
    # encoder_name = 'SynCLR_proteus_vit_base_14'
    # encoder_name = 'CLIP_proteus_vit_base_14'
    # encoder_name = 'SynCLR_vit_base_16'

    # interested_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    # interested_layers = [2, 5, 8, 11]
    # interested_layers = [11]
    interested_layers = list(range(4, 19))
    # fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise "Architecture not in small, base, large."

    model = DADE(encoder=encoder, interested_layers=interested_layers,
                 rotate_supports=True, use_vv_attn=False)
    model = model.to(device)

    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

    for item, test_data, support_data in zip(test_item_list, test_data_list, support_data_list):
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                      num_workers=0)#4
        seed_generator = torch.Generator()
        seed_generator.manual_seed(args.seed)
        support_loader = torch.utils.data.DataLoader(support_data, batch_size=args.shots, shuffle=True,
                                                     num_workers=0, generator=seed_generator)
        results = evaluation_batch_vis(model, test_dataloader, support_loader, device, max_ratio=0.01,
                                   resize_mask=256)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

        auroc_sp_list.append(auroc_sp)
        ap_sp_list.append(ap_sp)
        f1_sp_list.append(f1_sp)
        auroc_px_list.append(auroc_px)
        ap_px_list.append(ap_px)
        f1_px_list.append(f1_px)
        aupro_px_list.append(aupro_px)

        print_fn(
            '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

    print_fn(
        'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
            np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_data_path', type=str, default='/data/disk8T2/guoj/Real-IAD')
    parser.add_argument('--test_data_path', type=str, default='/mnt/disk2/wangqs/Datasets/AdaCLIP_Datasets/mvtec_anomaly_detection')
    parser.add_argument('--shots', type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='test')
    args = parser.parse_args()

    # train_item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
    #                    'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
    #                    'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
    #                    'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
    #                    'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    
    test_item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                      'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    print_fn("shots: %d" % args.shots)
    print_fn("seed: %d" % args.seed)
    train()

# python dade_mvtec_test.py --shots 1 --seed 1