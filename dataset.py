import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json
import albumentations as alb
from albumentations.pytorch import ToTensorV2
# import datasets
import cv2
from perlin import rand_perlin_2d_np
import scipy.ndimage as ndimage
from skimage import measure

import matplotlib.pyplot as plt

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


def get_strong_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    return data_transforms


def get_alb_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = alb.Compose([alb.Resize(size, size),
                                   # alb.HorizontalFlip(p=0.5),
                                   # alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5,
                                   #                      border_mode=cv2.BORDER_CONSTANT),
                                   alb.RandomResizedCrop(isize, isize, scale=(0.65, 1), ratio=(0.9, 1.1)),
                                   alb.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5),
                                   alb.Normalize(mean=mean_train, std=std_train),
                                   ToTensorV2(transpose_mask=True)
                                   ])
    return data_transforms


def get_strong_alb_transforms(size, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = alb.Compose([alb.Resize(size, size),
                                   alb.HorizontalFlip(p=0.5),
                                   alb.ElasticTransform(p=0.5, alpha=1, sigma=20, alpha_affine=20,
                                                        border_mode=cv2.BORDER_CONSTANT),
                                   alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=(-0.1, 0.2), rotate_limit=15, p=1.,
                                                        border_mode=cv2.BORDER_CONSTANT),
                                   alb.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.),
                                   alb.Normalize(mean=mean_train, std=std_train),
                                   ToTensorV2(transpose_mask=True)
                                   ])
    return data_transforms


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            a = 1
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class LOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*/000.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        size = (img.size[1], img.size[0])
        img = self.transform(img)
        type = self.types[idx]
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path, type, size


class INDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, shots=4):

        self.img_path = root
        self.transform = transform
        self.shots = shots
        self.img_paths = glob.glob(self.img_path + "/*.JPEG")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.img_paths[idx]
            img = Image.open(img_path).convert('RGB')
        except:
            img_path = self.img_paths[idx - 1]
            img = Image.open(img_path).convert('RGB')
        query_img = self.transform(img)

        support_img = []
        for i in range(self.shots):
            support_img.append(self.transform(img))
        support_img = torch.stack(support_img, dim=0)

        return query_img, support_img


class INDRAEMDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, shots=4):

        self.img_path = root
        self.transform = transform
        self.shots = shots
        self.img_paths = glob.glob(self.img_path + "/*.JPEG")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.img_paths[idx]
            img = Image.open(img_path).convert('RGB')
        except:
            img_path = self.img_paths[idx - 1]
            img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        query_img = self.transform(image=img)['image']
        size = query_img.shape[1]

        perlin_scale = 6
        min_perlin_scale = 1

        anomaly_source_idx = torch.randint(0, len(self.img_paths), (1,)).item()
        anomaly_source_img = Image.open(self.img_paths[anomaly_source_idx]).convert('RGB')
        anomaly_source_img = np.array(anomaly_source_img)
        anomaly_source_img = self.transform(image=anomaly_source_img)['image']

        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((256, 256), (perlin_scalex, perlin_scaley))
        perlin_noise = cv2.resize(perlin_noise, dsize=(size, size))
        # perlin_noise = scipy.ndimage.rotate(perlin_noise, angle=torch.randint(low=-90, high=90, size=(1,))[0].item(),
        #                                   order=2, reshape=False)

        threshold = 0.5
        perlin_thr = perlin_noise > threshold
        perlin_thr = self.process_binary_mask(perlin_thr, min_pixels=20,
                                              n_regions=torch.randint(1, 5, (1,)).item())

        perlin_thr = torch.tensor(perlin_thr).unsqueeze(0).float()
        anomaly_source_img_thr = anomaly_source_img * perlin_thr

        beta = random.random() * 0.5
        augmented_img = query_img * (1 - perlin_thr) + (
                1 - beta) * anomaly_source_img_thr + beta * query_img * perlin_thr

        support_img = []
        for i in range(self.shots):
            support_img.append(self.transform(image=img)['image'])
        support_img = torch.stack(support_img, dim=0)

        query_gt = torch.zeros([1, query_img.size()[-2], query_img.size()[-2]])
        augmented_gt = perlin_thr
        return augmented_img, augmented_gt, query_img, query_gt, support_img

    def process_binary_mask(self, binary_image, min_pixels=10, n_regions=5):
        # 标记连通域
        labeled_image, num_features = ndimage.label(binary_image)

        # 计算每个连通域的属性
        regions = measure.regionprops(labeled_image)

        # 过滤出大于等于min_pixels的连通域
        valid_regions = [region for region in regions if region.area >= min_pixels]

        # 如果有效连通域数量少于n，则全部保留
        if len(valid_regions) <= n_regions:
            selected_regions = valid_regions
        else:
            # 随机选择n个连通域
            selected_regions = random.sample(valid_regions, n_regions)

        # 创建一个新的标签图像，只保留选中的连通域
        filtered_labels = np.zeros_like(labeled_image)
        for region in selected_regions:
            filtered_labels[labeled_image == region.label] = 1

        return filtered_labels


class MiniDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):

        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
            img_tot_paths.extend(img_paths)
            tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path, label = self.img_paths[idx], self.labels[idx]
            img = Image.open(img_path).convert('RGB')
        except:
            img_path, label = self.img_paths[idx - 1], self.labels[idx - 1]
            img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label


class RealIADDatasetFADE(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, normal_shots=4, same_view=True):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.normal_shots = normal_shots
        self.same_view = same_view

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths_good, self.good_view_num = [], []
        self.img_paths_bad, self.gt_paths_bad, self.bad_view_num = [], [], []

        self.img_paths_good_view = [[], [], [], [], []]
        for phase in ['train', 'test']:
            data_set = class_json[phase]
            for sample in data_set:
                label = sample['anomaly_class'] != 'OK'
                if label:
                    view_num = self.get_view(sample['image_path'])
                    self.img_paths_bad.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
                    self.gt_paths_bad.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
                    self.bad_view_num.append(view_num)
                else:
                    view_num = self.get_view(sample['image_path'])
                    self.img_paths_good.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
                    self.good_view_num.append(view_num)
                    self.img_paths_good_view[view_num].append(
                        os.path.join(root, 'realiad_1024', category, sample['image_path']))

        self.img_paths_good = np.array(self.img_paths_good)
        self.good_view_num = np.array(self.good_view_num)
        self.img_paths_good_view = np.array(self.img_paths_good_view, dtype=object)

        self.img_paths_bad = np.array(self.img_paths_bad)
        self.gt_paths_bad = np.array(self.gt_paths_bad)
        self.bad_view_num = np.array(self.bad_view_num)

        self.cls_idx = 0

    def get_view(self, file_name):
        view = file_name.split('_')[-2]
        assert 'C' in view
        view_num = int(view[1]) - 1
        return view_num

    def __len__(self):
        return len(self.img_paths_bad)

    def __getitem__(self, idx):

        img_path, gt_path = self.img_paths_bad[idx], self.gt_paths_bad[idx]
        bad_img = Image.open(img_path).convert('RGB')
        bad_img = np.array(bad_img)
        bad_gt = Image.open(gt_path)
        bad_gt = np.expand_dims(np.array(bad_gt), axis=-1)

        out = self.transform(image=bad_img, mask=bad_gt)
        bad_img, bad_gt = out['image'], out['mask']
        bad_gt = bad_gt / 255

        assert bad_img.size()[1:] == bad_gt.size()[1:], "image.size != gt.size !!!"

        view_num = self.bad_view_num[idx]

        if self.same_view:
            img_paths_good = self.img_paths_good_view[view_num]
        else:
            img_paths_good = self.img_paths_good

        support_idx = random.sample(range(len(img_paths_good)), self.normal_shots)
        support_img = []
        for idx in support_idx:
            img = Image.open(img_paths_good[idx]).convert('RGB')
            img = np.array(img)
            img = self.transform(image=img)['image']
            support_img.append(img)
        support_img = torch.stack(support_img, dim=0)

        good_idx = random.sample(range(len(img_paths_good)), 1)
        img = Image.open(img_paths_good[good_idx[0]]).convert('RGB')
        img = np.array(img)
        good_img = self.transform(image=img)['image']
        good_gt = torch.zeros([1, good_img.size()[-2], good_img.size()[-2]])

        return bad_img, bad_gt, good_img, good_gt, support_img


class MVTecDatasetFADE(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, normal_shots=4, same_view=True):
        self.img_path = os.path.join(root, category)
        self.transform = transform
        self.normal_shots = normal_shots

        self.img_paths_good = []
        self.img_paths_bad, self.gt_paths_bad = [], []

        for phase in ['train', 'test']:
            defect_types = os.listdir(os.path.join(self.img_path, phase))

            for defect_type in defect_types:
                if defect_type == 'good':
                    img_paths = glob.glob(os.path.join(self.img_path, phase, defect_type) + "/*.png") + \
                                glob.glob(os.path.join(self.img_path, phase, defect_type) + "/*.JPG") + \
                                glob.glob(os.path.join(self.img_path, phase, defect_type) + "/*.bmp")
                    img_paths.sort()
                    self.img_paths_good.extend(img_paths)
                else:
                    img_paths = glob.glob(os.path.join(self.img_path, phase, defect_type) + "/*.png") + \
                                glob.glob(os.path.join(self.img_path, phase, defect_type) + "/*.JPG") + \
                                glob.glob(os.path.join(self.img_path, phase, defect_type) + "/*.bmp")
                    gt_paths = glob.glob(os.path.join(self.img_path, 'ground_truth', defect_type) + "/*.png")
                    img_paths.sort()
                    gt_paths.sort()
                    self.img_paths_bad.extend(img_paths)
                    self.gt_paths_bad.extend(gt_paths)

        self.img_paths_good = np.array(self.img_paths_good)

        self.img_paths_bad = np.array(self.img_paths_bad)
        self.gt_paths_bad = np.array(self.gt_paths_bad)

        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths_bad)

    def __getitem__(self, idx):

        img_path, gt_path = self.img_paths_bad[idx], self.gt_paths_bad[idx]
        bad_img = Image.open(img_path).convert('RGB')
        bad_img = np.array(bad_img)
        bad_gt = Image.open(gt_path)
        bad_gt = np.expand_dims(np.array(bad_gt), axis=-1)

        out = self.transform(image=bad_img, mask=bad_gt)
        bad_img, bad_gt = out['image'], out['mask']
        bad_gt = bad_gt / 255

        assert bad_img.size()[1:] == bad_gt.size()[1:], "image.size != gt.size !!!"

        support_idx = random.sample(range(len(self.img_paths_good)), self.normal_shots)
        support_img = []
        for idx in support_idx:
            img = Image.open(self.img_paths_good[idx]).convert('RGB')
            img = np.array(img)
            img = self.transform(image=img)['image']
            support_img.append(img)
        support_img = torch.stack(support_img, dim=0)

        good_idx = random.sample(range(len(self.img_paths_good)), 1)
        img = Image.open(self.img_paths_good[good_idx[0]]).convert('RGB')
        img = np.array(img)
        good_img = self.transform(image=img)['image']
        good_gt = torch.zeros([1, good_img.size()[-2], good_img.size()[-2]])

        return bad_img, bad_gt, good_img, good_gt, support_img


class MVTecDatasetFS(torch.utils.data.Dataset):
    def __init__(self, root, transform, shots=4, seed=0):
        self.img_path = os.path.join(root, 'train')
        self.transform = transform
        self.shots = shots
        # load dataset
        self.img_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1

        rng = np.random.Generator(np.random.PCG64(seed=seed))
        random_idx = rng.choice(len(self.img_paths), size=shots, replace=False)
        self.img_paths = self.img_paths[random_idx]
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)

        return np.array(img_tot_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)

        return img, img_path
