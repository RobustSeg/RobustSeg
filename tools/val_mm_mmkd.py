import torch
import argparse
import yaml
import math
import os
import time
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from math import ceil
import numpy as np
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from semseg.models.segformer.seg_block_select_UMD import Seg
# from semseg.models.segformer.seg_block_UMDt import Seg


def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img


@torch.no_grad()
def sliding_predict(model, image, num_classes, flip=True):
    image_size = image[0].shape
    tile_size = (int(ceil(image_size[2] * 1)), int(ceil(image_size[3] * 1)))
    overlap = 1 / 3

    stride = ceil(tile_size[0] * (1 - overlap))

    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = torch.zeros((num_classes, image_size[2], image_size[3]), device=torch.device('cuda'))
    count_predictions = torch.zeros((image_size[2], image_size[3]), device=torch.device('cuda'))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = [modal[:, :, y_min:y_max, x_min:x_max] for modal in image]
            padded_img = [pad_image(modal, tile_size) for modal in img]
            tile_counter += 1
            padded_prediction = model(padded_img)
            if flip:
                fliped_img = [padded_modal.flip(-1) for padded_modal in padded_img]
                fliped_predictions = model(fliped_img)
                padded_prediction += fliped_predictions.flip(-1)
            predictions = padded_prediction[:, :, :img[0].shape[2], :img[0].shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.squeeze(0)

    return total_predictions.unsqueeze(0)


@torch.no_grad()
def evaluate(model, dataloader, device):
    print('Evaluating...')
    model.eval()
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
    sliding = False
    for images, labels in tqdm(dataloader):
        images = [x.to(device) for x in images]
        labels = labels.to(device)
        if sliding:
            preds = sliding_predict(model, images, num_classes=n_classes).softmax(dim=1)
        else:
            preds = model(images)[0].softmax(dim=1)
        metrics.update(preds, labels)

    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()

    return acc, macc, f1, mf1, ious, miou

@torch.no_grad()
def evaluate_zero_padding(model, dataloader, device, modalities, dataset_name):
    print(f'Evaluating...{modalities}')
    all_modalities = []
    if dataset_name == "DELIVER":
        all_modalities = ['RGB', 'D', 'E', 'L']
    elif dataset_name == "MUSES":
        all_modalities = ['F', 'E', 'L']
    elif dataset_name == "MCubeS":
        all_modalities = ['image', 'aolp', 'dolp', 'nir']

    missing_modalities = [md for md in all_modalities if md not in modalities]
    model.eval()
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
    sliding = False
    for images, labels in tqdm(dataloader):
        images = [x.to(device) for x in images]
        labels = labels.to(device)

        for modality in missing_modalities:
            index = all_modalities.index(modality)
            images[index].zero_()

        if sliding:
            preds = sliding_predict(model, images, num_classes=n_classes).softmax(dim=1)
        else:
            preds = model(images)[0].softmax(dim=1)
        metrics.update(preds, labels)

    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()

    return acc, macc, f1, mf1, ious, miou


@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = [F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=True) for img in
                             images]
            scaled_images = [scaled_img.to(device) for scaled_img in scaled_images]
            logits = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = [torch.flip(scaled_img, dims=(3,)) for scaled_img in scaled_images]
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)

    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou

def evaluate_model(model, dataloader, device, eval_cfg, dataset, modals_tag):
    """
    评估模型的函数，根据配置进行单尺度或多尺度评估。

    参数:
        dataloader (DataLoader): 数据加载器，用于加载待评估的数据。
        model (nn.Module): 待评估的模型。
        device (torch.device): 设备信息（CPU或GPU）。
        eval_cfg (dict): 包含评估配置的字典，包括MSF配置。
        dataset_idel (module): 数据集相关类或模块，包含类别信息。
        save_results (callable): 保存结果的回调函数或可执行代码。

    返回:
        float: 均值交并比（mIoU）。
    """
    modals = ''.join([m[0] for m in modals_tag])
    # 根据配置选择评估方式
    if eval_cfg['MSF']['ENABLE']:
        acc, macc, f1, mf1, ious, miou = evaluate_msf(
            model,
            dataloader,
            device,
            eval_cfg['MSF']['SCALES'],
            eval_cfg['MSF']['FLIP']
        )
    else:
        acc, macc, f1, mf1, ious, miou = evaluate(
            model,
            dataloader,
            device
        )

    # 构建结果表格
    table = {
        'Class': list(dataset.CLASSES) + ['Mean'],
        'IoU': ious + [miou],
        'F1': f1 + [mf1],
        'Acc': acc + [macc]
    }

    # 打印和保存结果
    print(f"Evaluating on {modals} mIoU : {miou}")

    return miou, table

def main(cfg):
    # 定义 idel 的模态
    idel = ['img', 'depth', 'event', 'lidar']
    n = len(idel)
    modal_groups = {}
    # 遍历所有可能的组合长度（从1到n）
    for r in range(1, n + 1):
        # 使用位运算生成所有可能的组合
        for bitmask in range(1, 1 << n):
            # 计算二进制中1的个数，确保符合当前组合长度
            if bin(bitmask).count('1') == r:
                # 提取选中的模态
                selected = [idel[i] for i in range(n) if (bitmask & (1 << i))]
                key = ''.join([item[0] for item in selected])
                modal_groups[key] = selected

    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    # cases = ['cloud', 'fog', 'night', 'rain', 'sun']
    # cases = ['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    cases = [None]  # all

    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists():
        raise FileNotFoundError
    print(f"Evaluating {model_path}...")

    # exp_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # eval_path = os.path.join(os.path.dirname(eval_cfg['MODEL_PATH']), 'eval_{}.txt'.format(exp_time))

    for case in cases:
        datasets = {}
        dataloaders = {}

        for key , modals in modal_groups.items():
            dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform, modals, case)
            datasets[f'dataset_{key}'] = dataset
            sampler_val = None
            # 创建数据加载器
            dataloader = DataLoader(
                dataset,
                batch_size=eval_cfg['BATCH_SIZE'],
                num_workers=8,
                pin_memory=False,
                sampler=sampler_val
            )
            dataloaders[f'dataloader_{key}'] = dataloader


        # dataset_idel = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform, cfg['DATASET']['MODALS'], case)
        #
        # dataset_id = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform, id, case)
        # dataset_ie = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform, ie, case)
        # dataset_il = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform, il, case)
        # dataset_idl = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform, idl, case)
        # dataset_ide = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform, ide, case)
        # --- test set
        # dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'test', transform, cfg['DATASET']['MODALS'], case)

        model = Seg(cfg['MODEL']['BACKBONE'], num_classes=datasets['dataset_idel'].n_classes, pretrained=True, modals=cfg['DATASET']['MODALS'])
        msg = model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        print(msg)
        model = model.to(device)
        model.eval()

        mIoU_dict = {}
        table_dict = {}

        for key , modals in modal_groups.items():
            dataloader_var = f'dataloader_{key}'
            dataset_var = f'dataset_{key}'

            mIoU, table = evaluate_model(model, dataloaders[dataloader_var], device, eval_cfg, datasets[dataset_var],
                                                   modals)

            # 存储结果
            mIoU_dict[f"mIoU_{key}"] = mIoU
            table_dict[f"table_{key}"] = table

        # tables = [table_idel, table_id, table_ie, table_il, table_ide, table_idl]
        # modals = ["idel", "id" ,"ie" ,"il", "ide", "idl"]

        mean_mIoU = 0.0
        exp_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        eval_path = os.path.join(os.path.dirname(eval_cfg['MODEL_PATH']), 'eval_{}.txt'.format(exp_time))
        with open(eval_path, 'a+') as f:
            for key, table in modal_groups.items():
                f.writelines(eval_cfg['MODEL_PATH'])
                f.write("\n============== Eval in {} modals =================\n".format(key))
                f.write("\n")
                print(tabulate(table_dict[f"table_{key}"], headers='keys'), file=f)
                mean_mIoU += mIoU_dict[f"mIoU_{key}"]
        mean_mIoU /= len(modal_groups)
        with open(eval_path, 'a+') as f:
            f.write("\n============== mean mIoU {} =================\n".format(mean_mIoU))
        print("Results saved around {}".format(eval_cfg['MODEL_PATH']))
        print(f"mean mIoU {mean_mIoU}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/DELIVER.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    # gpu = setup_ddp()
    # main(cfg, gpu)
    main(cfg)