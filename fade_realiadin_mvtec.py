# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from dataset import get_data_transforms, get_alb_transforms, get_strong_data_transforms, get_strong_alb_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset

from models.uad import FADE, FADEv3
from models import vit_encoder
from torch.nn.init import trunc_normal_
from models.vision_transformer import Block as VitBlock, Attention, CrossAttention, CrossBlock, Mlp, VVCrossAttention
from dataset import MVTecDataset, RealIADDatasetFADE, INDRAEMDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, WarmCosineScheduler, FocalBCE, DiceLossBatch, BCE, distance_loss
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
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
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
    setup_seed(1)

    imagenet_iters = 10000
    total_iters = 20000
    batch_size = 8
    image_size = 392
    crop_size = 392

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    train_transform = get_alb_transforms(image_size)
    realiad_data = []
    """Train Data"""
    for i, item in enumerate(train_item_list):
        train_data = RealIADDatasetFADE(root=args.realiad_data_path, category=item, transform=train_transform,
                                        normal_shots=args.shots, same_view=True)
        realiad_data.append(train_data)
    realiad_data = ConcatDataset(realiad_data)

    strong_data_transform = get_strong_alb_transforms(image_size)
    imagenet_data = INDRAEMDataset(root=args.imagenet_data_path, transform=strong_data_transform, shots=args.shots)

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
    encoder_name = 'dinov2reg_vit_base_14'
    # encoder_name = 'dinov2reg_vit_large_14'

    interested_layers = [2, 3, 4, 5, 6, 7, 8, 9]

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise "Architecture not in small, base, large."

    bottleneck = Mlp(embed_dim, embed_dim * 1, embed_dim)

    decoder = []
    for i in range(4):
        blk = CrossBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=1.,
                         qkv_bias=False, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0., drop=0.,
                         attn=CrossAttention, support_norm=True, init_values=None)
        decoder.append(blk)
        # blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=1.,
        #                qkv_bias=False, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.,
        #                attn=Attention, init_values=1e-6)
        # bottleneck.append(blk)
    decoder = nn.ModuleList(decoder)

    head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    model = FADEv3(encoder=encoder, bottleneck=bottleneck, decoder=decoder, head=head,
                   interested_layers=interested_layers, num_anomaly_registers=0, remove_class_token=False)
    model = model.to(device)

    optimizer = torch.optim.AdamW([{'params': model.bottleneck.parameters()},
                                   {'params': model.decoder.parameters()},
                                   {'params': model.head.parameters()},
                                   ],
                                  lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, amsgrad=False, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-4, final_value=1e-4,
                                       total_iters=total_iters + imagenet_iters,
                                       warmup_iters=1000)

    focal_loss = FocalBCE(gamma=2, alpha=5)

    """
    ImageNet Training
    """
    train_dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=batch_size, shuffle=True, num_workers=8,
                                                   drop_last=True)
    print_fn('imagenet image number:{}'.format(len(imagenet_data)))
    it = 0
    for epoch in range(int(np.ceil(imagenet_iters / len(train_dataloader)))):
        model.train()

        loss_list = []
        for bad_img, bad_gt, good_img, good_gt, support_img in train_dataloader:
            support_img = support_img.to(device)
            query_img = torch.cat([bad_img, good_img]).to(device)
            gt = torch.cat([bad_gt, good_gt]).to(device)
            pred, dist = model(query_img, support_img)

            gt = torch.max_pool2d(gt, kernel_size=14, stride=14)
            loss = 5 * focal_loss(pred, gt) + distance_loss(dist, gt)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(bottleneck.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            if (it + 1) == imagenet_iters:
                break
            if (it + 1) % 100 == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, imagenet_iters, np.mean(loss_list)))
                loss_list = []
            it += 1

    """
    Real-IAD Training
    """
    train_dataloader = torch.utils.data.DataLoader(realiad_data, batch_size=batch_size, shuffle=True, num_workers=8,
                                                   drop_last=True)
    print_fn('realiad image number:{}'.format(len(realiad_data)))
    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()

        loss_list = []
        for bad_img, bad_gt, good_img, good_gt, support_img in train_dataloader:
            support_img = support_img.to(device)
            query_img = torch.cat([bad_img, good_img]).to(device)
            gt = torch.cat([bad_gt, good_gt]).to(device)
            pred, dist = model(query_img, support_img)

            gt = torch.max_pool2d(gt, kernel_size=14, stride=14)
            loss = 5 * focal_loss(pred, gt) + distance_loss(dist, gt)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(bottleneck.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            # if (it + 1) % 10000 == 0:
            #     torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

            if (it + 1) % 5000 == 0:
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data, support_data in zip(test_item_list, test_data_list, support_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                  num_workers=4)
                    support_loader = torch.utils.data.DataLoader(support_data, batch_size=args.shots, shuffle=True,
                                                                 num_workers=0)
                    results = evaluation_batch(model, test_dataloader, support_loader, device, max_ratio=0.01,
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

                model.train()

            if (it + 1) == total_iters:
                break
            if (it + 1) % 100 == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []
            it += 1

    # torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--realiad_data_path', type=str, default='/data/disk8T2/guoj/Real-IAD')
    parser.add_argument('--imagenet_data_path', type=str, default='/data/disk8T2/guoj/imagenet/testset')
    parser.add_argument('--test_data_path', type=str, default='../mvtec_anomaly_detection')
    parser.add_argument('--shots', type=int, default=4)

    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='fadev4_inreal_s4sv_mvtec_dinov2br_fs8l_saug_ca4_star_it20k_adam1e4_w1k1w2w_mp14_5fo5dist_b8_s1')
    args = parser.parse_args()

    train_item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                       'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                       'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                       'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                       'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']

    test_item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                      'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    train()
