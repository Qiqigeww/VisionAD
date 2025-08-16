import torch, logging, os, tabulate, datetime
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader

from torch.nn import functional as F
from sklearn import metrics
from sklearn.metrics import roc_auc_score, auc, average_precision_score, precision_recall_curve
import cv2, copy
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
# from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle


class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )


class EvalDataMeta:
    def __init__(self, preds, masks):
        self.preds = preds  # N x H x W
        self.masks = masks  # N x H x W


class EvalImage:
    def __init__(self, data_meta, **kwargs):
        self.preds = self.encode_pred(data_meta.preds, **kwargs)
        self.masks = self.encode_mask(data_meta.masks)
        self.preds_good = sorted(self.preds[self.masks == 0], reverse=True)
        self.preds_defe = sorted(self.preds[self.masks == 1], reverse=True)
        self.num_good = len(self.preds_good)
        self.num_defe = len(self.preds_defe)

    @staticmethod
    def encode_pred(preds):
        raise NotImplementedError

    def encode_mask(self, masks):
        N, _, _ = masks.shape
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(int)  # (N, )
        return masks

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


class EvalImageMean(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).mean(axis=1)  # (N, )


class EvalImageStd(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).std(axis=1)  # (N, )


class EvalImageMax(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        for i in range(0, 8):
            preds = (F.avg_pool2d(preds, 8, stride=1))
        preds = preds.cpu().numpy()  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )


class EvalPerPixelAUC:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc

class EvalPerPixelPRO:
    def __init__(self, data_meta):
        self.preds = data_meta.preds
        self.masks = data_meta.masks
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        pro = compute_pro(self.masks, self.preds)
        return pro

class EvalPerPixelAP:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1
    def eval_auc(self):
        ap = average_precision_score(self.masks, self.preds)
        return ap
    
class EvalImageAP(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        for i in range(0, 8):
            preds = (F.avg_pool2d(preds, 8, stride=1))
        preds = preds.cpu().numpy()  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )
    def eval_auc(self):
        ap = average_precision_score(self.masks, self.preds)
        return ap
    
class EvalPerPixelF1:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1
    def eval_auc(self):
        precisions, recalls, thresholds = precision_recall_curve(self.masks, self.preds)
        epsilon = 1e-8
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + epsilon)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        return f1_px
    
class EvalImageF1(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        for i in range(0, 8):
            preds = (F.avg_pool2d(preds, 8, stride=1))
        preds = preds.cpu().numpy()  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )
    def eval_auc(self):
        precisions, recalls, thresholds = precision_recall_curve(self.masks, self.preds)
        epsilon = 1e-8
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + epsilon)
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
        return f1_sp
    
class EvalPerPixelAUPR:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1
    def eval_auc(self):
        pr_auc = compute_aupr(self.preds, self.masks)
        return pr_auc
    
class EvalImageAUPR(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        for i in range(0, 8):
            preds = (F.avg_pool2d(preds, 8, stride=1))
        preds = (F.avg_pool2d(preds, 2, stride=1))
        preds = preds.cpu().numpy()  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )
    def eval_auc(self):
        pr_auc = compute_aupr(self.preds, self.masks)
        return pr_auc

eval_lookup_table = {
    "mean": EvalImageMean,
    "std": EvalImageStd,
    "max": EvalImageMax,
    "pixel": EvalPerPixelAUC,
    "pro" :EvalPerPixelPRO,
    "appx": EvalPerPixelAP,
    "apsp": EvalImageAP,
    "f1px": EvalPerPixelF1,
    "f1sp": EvalImageF1,
    "auprpx": EvalPerPixelAUPR,
    "auprsp": EvalImageAUPR,
}

def performances(gt_list_px, pr_list_px, clsname_list, args):
    pr_list_px = np.concatenate(np.asarray(pr_list_px), axis=0)
    gt_list_px = np.concatenate(np.asarray(gt_list_px), axis=0)
    ret_metrics = {}
    clsnames = set(clsname_list)
    for clsname in clsnames:
        preds_cls = []
        masks_cls = []
        for filename, pred, mask in zip(clsname_list, pr_list_px, gt_list_px):
            if filename == clsname:
                preds_cls.append(pred[None, ...])
                masks_cls.append(mask[None, ...])
        preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)
        masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)
        data_meta = EvalDataMeta(preds_cls, masks_cls)
        
        for metric in args.metrics:
            eval_method = eval_lookup_table[metric](data_meta)
            auc = eval_method.eval_auc()
            ret_metrics[f"{clsname}_{metric}_auc"] = round(auc, 3)
        
    for metric in args.metrics:
        evalvalues = [
            ret_metrics[f"{clsname}_{metric}_auc"]
            for clsname in clsnames
        ]
        mean_auc = np.mean(np.array(evalvalues))
        ret_metrics["{}_{}_auc".format("mean", metric)] = round(mean_auc, 3)                      
        
    return ret_metrics

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def log_metrics(ret_metrics):
    logger = logging.getLogger("global_logger")
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"])) + ["mean"]

    # auc
    auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
    evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
    record = Report(["clsname"] + evalnames)

    for clsname in clsnames:
        clsvalues = [
            ret_metrics["{}_{}_auc".format(clsname, evalname)]
            for evalname in evalnames
        ]
        record.add_one_record([clsname] + clsvalues)
    
    logger.info(f"\n{record}\n")


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
        df_new = pd.DataFrame([{"pro": mean(pros), "fpr": fpr, "threshold": th}])
        df = pd.concat([df, df_new], ignore_index=True)
        # df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()
    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def compute_aupr(
    predicted_masks,
    ground_truth_masks,
    include_optimal_threshold_rates=False,
):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        predicted_masks: [list of np.arrays or np.array] [NxHxW] Contains
                               generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    pred_mask = copy.deepcopy(predicted_masks)
    gt_mask = copy.deepcopy(ground_truth_masks)
    num = 200
    out = {}

    if pred_mask is None or gt_mask is None:
        for key in out:
            out[key].append(float('nan'))
    else:
        fprs, tprs = [], []
        precisions, f1s = [], []
        gt_mask = np.array(gt_mask, np.uint8)

        t = (gt_mask == 1)
        f = ~t
        n_true = t.sum()
        n_false = f.sum()
        th_min = pred_mask.min() - 1e-8
        th_max = pred_mask.max() + 1e-8
        pred_gt = pred_mask[t]
        th_gt_min = pred_gt.min()
        th_gt_max = pred_gt.max()

        '''
        Using scikit learn to compute pixel au_roc results in a memory error since it tries to store the NxHxW float score values.
        To avoid this, we compute the tp, fp, tn, fn at equally spaced thresholds in the range between min of predicted 
        scores and maximum of predicted scores
        '''
        percents = np.linspace(100, 0, num=num // 2)
        th_gt_per = np.percentile(pred_gt, percents)
        th_unif = np.linspace(th_gt_max, th_gt_min, num=num // 2)
        thresholds = np.concatenate([th_gt_per, th_unif, [th_min, th_max]])
        thresholds = np.flip(np.sort(thresholds))

        if n_true == 0 or n_false == 0:
            raise ValueError("gt_submasks must contains at least one normal and anomaly samples")

        for th in thresholds:
            p = (pred_mask > th).astype(np.uint8)
            p = (p == 1)
            fp = (p & f).sum()
            tp = (p & t).sum()

            fpr = fp / n_false
            tpr = tp / n_true
            if tp + fp > 0:
                prec = tp / (tp + fp)
            else:
                prec = 1.0
            if prec > 0. and tpr > 0.:
                f1 = (2 * prec * tpr) / (prec + tpr)
            else:
                f1 = 0.0
            fprs.append(fpr)
            tprs.append(tpr)
            precisions.append(prec)

        pr_auc = metrics.auc(tprs, precisions)
        pr_auc = round(pr_auc, 4)

    return pr_auc

def load_checkpoint_if_exists(ckp_path, decoder, bn):
    if os.path.isfile(ckp_path):
        print("=> loading checkpoint '{}'".format(ckp_path))
        ckp = torch.load(ckp_path)

        # 移除不需要的 'memory' 键
        for k in list(ckp['bn'].keys()):
            if 'memory' in k:
                print("Removing memory key from batch normalization: {}".format(k))
                ckp['bn'].pop(k)

        # 加载 decoder 和 batch normalization 的状态
        decoder.load_state_dict(ckp['decoder'], strict=False)
        bn.load_state_dict(ckp['bn'], strict=False)
        
        print("=> successfully loaded checkpoint '{}'".format(ckp_path))
    else:
        print("=> no checkpoint found at '{}', skipping load".format(ckp_path))
        
 #  原来的各图叠放到一起的;
def visualize_compound(img_path_list, pr_list_px, gt_list_px, vis_dir):
       
    max_score = pr_list_px.max()
    min_score = pr_list_px.min()

    for i, (img_path, pr_px, gt_px) in enumerate(zip(img_path_list, pr_list_px, gt_list_px)):
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        height, width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filedir, filename = os.path.split(img_path)
        _, defename = os.path.split(filedir)
        clsname = img_path.split('/')[-4]
        save_dir = os.path.join(vis_dir, clsname, defename)
        os.makedirs(save_dir, exist_ok=True)

        # read image
        h, w = int(height), int(width)
        pred = pr_px[:, :, None].repeat(3, 2)
        pred = cv2.resize(pred, (w, h))

        # self normalize just for analysis
        scoremap_self = apply_ad_scoremap(image, normalize(pred))
        # global normalize
        pred = np.clip(pred, min_score, max_score)
        pred = normalize(pred, max_score, min_score)
        scoremap_global = apply_ad_scoremap(image, pred)

        # if gt_px is not None:
        mask = (gt_px * 255).astype(np.uint8)[:, :, None].repeat(3, 2)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        save_path = os.path.join(save_dir, filename)
        if mask.sum() == 0:
            scoremap = np.vstack([image, scoremap_global])
        else:
            scoremap = np.vstack([image, mask, scoremap_global, scoremap_self])

        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, scoremap)
        # print('We have saved the image:', img_path)    
        
def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float64)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)
                 
def setup_logger():
    # 日志配置
    logger_name = "global_logger"
    
    if not os.path.exists("./logs"):  # 如果目录不存在，则创建
        os.makedirs("./logs")
            
    log_file = "./logs/dec_{}.log".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    level = logging.INFO

    log = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s"
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fh)
    log.addHandler(sh)
    
    return log

# 各个图单独保存;
def visualize_compound_1(img_path_list, pr_list_px, gt_list_px, vis_dir):
    max_score = pr_list_px.max()
    min_score = pr_list_px.min()

    for i, (img_path, pr_px, gt_px) in enumerate(zip(img_path_list, pr_list_px, gt_list_px)):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        filedir, filename = os.path.split(img_path)
        basename, ext = os.path.splitext(filename)
        _, defename = os.path.split(filedir)
        clsname = img_path.split('/')[-4]
        save_dir = os.path.join(vis_dir, clsname, defename)
        os.makedirs(save_dir, exist_ok=True)

        h, w = int(height), int(width)
        pred = pr_px[:, :, None].repeat(3, 2)
        pred = cv2.resize(pred, (w, h))

        scoremap_self = apply_ad_scoremap(image_rgb, normalize(pred))
        pred = np.clip(pred, min_score, max_score)
        pred = normalize(pred, max_score, min_score)
        scoremap_global = apply_ad_scoremap(image_rgb, pred)

        # 处理GT mask
        mask = (gt_px * 255).astype(np.uint8)[:, :, None].repeat(3, 2)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if mask.sum() > 0:  # 异常图像才处理
            # 保存 ori、gt、global、local 图
            save_path_ori = os.path.join(save_dir, f"{basename}_ori{ext}")
            save_path_gt = os.path.join(save_dir, f"{basename}_gt{ext}")
            save_path_global = os.path.join(save_dir, f"{basename}_global{ext}")
            save_path_local = os.path.join(save_dir, f"{basename}_local{ext}")
            save_path_gt_red = os.path.join(save_dir, f"{basename}_gt_red{ext}")

            cv2.imwrite(save_path_ori, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_path_gt, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_path_global, cv2.cvtColor(scoremap_global, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_path_local, cv2.cvtColor(scoremap_self, cv2.COLOR_RGB2BGR))

            # 处理 gt_red 图像（先 resize mask）
            gt_resized = cv2.resize(gt_px.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bool = gt_resized > 0.5

            gt_red_image = image_rgb.copy()
            gt_red_image[mask_bool] = [255, 0, 0]  # 红色标注区域

            cv2.imwrite(save_path_gt_red, cv2.cvtColor(gt_red_image, cv2.COLOR_RGB2BGR))