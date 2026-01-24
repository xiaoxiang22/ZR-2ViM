## 2026/1/14 新增边界指标BF1
import os
import csv
import argparse
import logging
from typing import Dict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from ZR2ViM import ZR2ViM_Seg
from dataset import Data

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denorm_image(img_chw: torch.Tensor) -> np.ndarray:
    x = img_chw.detach().cpu().numpy()
    x = (x * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]) * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x


def _colorize_mask(mask_hw: np.ndarray, num_classes: int) -> np.ndarray:
    if num_classes == 2:
        gray = (mask_hw.astype(np.uint8) * 255)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    palette = [
        (0, 0, 0), (255, 255, 255), (0, 0, 255), (0, 255, 0),
        (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    vis = np.zeros((*mask_hw.shape, 3), dtype=np.uint8)
    for c in range(num_classes):
        vis[mask_hw == c] = palette[c % len(palette)]
    return vis


def save_triptych(img_chw, gt_hw, pred_hw, num_classes, save_path):
    img_vis = _denorm_image(img_chw)
    gt_vis = _colorize_mask(gt_hw.cpu().numpy(), num_classes)
    pred_vis = _colorize_mask(pred_hw.cpu().numpy(), num_classes)
    trip = np.concatenate([img_vis, gt_vis, pred_vis], axis=1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, trip)


def compute_boundary_f1(pred: np.ndarray, gt: np.ndarray, tau: int = 2) -> float:
    thr = 14

    pred = (pred.astype(np.uint8) * 255)
    gt = (gt.astype(np.uint8) * 255)

    contours_pd, _ = cv2.findContours(pred, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_gt, _ = cv2.findContours(gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours_gt and not contours_pd:
        return 1.0
    if not contours_gt or not contours_pd:
        return 0.0

    contours_gt_list = list()
    contours_pd_list = list()

    for i in range(len(contours_gt)):
        for j in range(len(contours_gt[i])):
            contours_gt_list.append(contours_gt[i][j][0].tolist())

    for i in range(len(contours_pd)):
        for j in range(len(contours_pd[i])):
            contours_pd_list.append(contours_pd[i][j][0].tolist())

    contour_gt_arr = np.array(contours_gt_list)
    contour_pd_arr = np.array(contours_pd_list)

    if contour_gt_arr.shape[0] == 0 or contour_pd_arr.shape[0] == 0:
        return 0.0

    num_matched_pd = 0
    num_matched_gt = 0
    num_gt = contour_gt_arr.shape[0]
    num_pd = contour_pd_arr.shape[0]

    for coor in contour_pd_arr:
        d2 = (contour_gt_arr[:, 0] - coor[0]) ** 2 + (contour_gt_arr[:, 1] - coor[1]) ** 2
        if np.min(d2) < thr ** 2:
            num_matched_pd += 1
    precision = num_matched_pd / (num_pd + 1e-7)

    for coor in contour_gt_arr:
        d2 = (contour_pd_arr[:, 0] - coor[0]) ** 2 + (contour_pd_arr[:, 1] - coor[1]) ** 2
        if np.min(d2) < thr ** 2:
            num_matched_gt += 1
    recall = num_matched_gt / (num_gt + 1e-7)
    return 2 * precision * recall / (precision + recall + 1e-7)


def build_model(args):
    return ZR2ViM_Seg(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depths=args.depths,
        decoder_depths=args.decoder_depths,
        num_heads=args.num_heads,
        window_size=args.window_size,
        drop_path_rate=args.drop_path_rate,
    )


def evaluate_test(model, loader, device, num_classes, outputs_dir, save_samples=10) -> Dict[str, float]:
    logger = logging.getLogger("test")
    model.eval()

    eps = 1e-7
    tp = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fp = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fn = torch.zeros(num_classes, dtype=torch.float64, device=device)

    total_pixels = 0
    correct_pixels = 0
    bf_scores = []

    total_infer_time = 0.0
    num_images = 0

    saved_vis = 0
    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(loader, desc="Testing")):
            images = batch['image'].float().to(device)
            masks = batch['label'].long().to(device).squeeze(1)
            names = batch.get('name', None)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            logits = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            total_infer_time += (t1 - t0)
            num_images += images.size(0)

            preds = torch.argmax(logits, dim=1)

            total_pixels += masks.numel()
            correct_pixels += (preds == masks).sum().item()

            for c in range(num_classes):
                tp[c] += ((preds == c) & (masks == c)).sum()
                fp[c] += ((preds == c) & (masks != c)).sum()
                fn[c] += ((preds != c) & (masks == c)).sum()

            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            for i in range(preds_np.shape[0]):
                bf_scores.append(
                    compute_boundary_f1(
                        pred=(preds_np[i] > 0),
                        gt=(masks_np[i] > 0),
                        tau=2
                    )
                )

            if saved_vis < save_samples:
                bs = images.size(0)
                for j in range(min(bs, save_samples - saved_vis)):
                    name = names[j] if names is not None else f"{bidx}_{j}"
                    out_path = os.path.join(outputs_dir, f"{name}_test.png")
                    save_triptych(images[j], masks[j], preds[j], num_classes, out_path)
                saved_vis += min(bs, save_samples - saved_vis)

    iou = (tp / (tp + fp + fn + eps)).mean().item()
    dice = ((2 * tp) / (2 * tp + fp + fn + eps)).mean().item()
    sen = (tp / (tp + fn + eps)).mean().item()
    spec = ((total_pixels - tp - fp - fn) / (total_pixels - tp - fn + eps)).mean().item()
    acc = correct_pixels / max(total_pixels, 1)
    bf = float(np.mean(bf_scores))

    infer_time_ms = (total_infer_time / max(num_images, 1)) * 1000.0

    logger.info(
        f"Test - Dice: {dice:.4f} | mIoU: {iou:.4f} | Acc: {acc:.4f} | "
        f"Sen: {sen:.4f} | Spe: {spec:.4f} | BF-score: {bf:.4f} | "
        f"Infer Time: {infer_time_ms:.2f} ms/img"
    )

    return dict(dice=dice, miou=iou, acc=acc, sen=sen, spe=spec, bf=bf, infer_ms=infer_time_ms)

    logger = logging.getLogger("test")
    model.eval()

    eps = 1e-7
    tp = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fp = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fn = torch.zeros(num_classes, dtype=torch.float64, device=device)

    total_pixels = 0
    correct_pixels = 0
    bf_scores = []

    saved_vis = 0
    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(loader, desc="Testing")):
            images = batch['image'].float().to(device)
            masks = batch['label'].long().to(device).squeeze(1)
            names = batch.get('name', None)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            total_pixels += masks.numel()
            correct_pixels += (preds == masks).sum().item()

            for c in range(num_classes):
                tp[c] += ((preds == c) & (masks == c)).sum()
                fp[c] += ((preds == c) & (masks != c)).sum()
                fn[c] += ((preds != c) & (masks == c)).sum()

            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            for i in range(preds_np.shape[0]):
                bf_scores.append(
                    compute_boundary_f1(
                        pred=(preds_np[i] > 0),
                        gt=(masks_np[i] > 0),
                        tau=2
                    )
                )

            if saved_vis < save_samples:
                bs = images.size(0)
                for j in range(min(bs, save_samples - saved_vis)):
                    name = names[j] if names is not None else f"{bidx}_{j}"
                    out_path = os.path.join(outputs_dir, f"{name}_test.png")
                    save_triptych(images[j], masks[j], preds[j], num_classes, out_path)
                saved_vis += min(bs, save_samples - saved_vis)

    iou = (tp / (tp + fp + fn + eps)).mean().item()
    dice = ((2 * tp) / (2 * tp + fp + fn + eps)).mean().item()
    sen = (tp / (tp + fn + eps)).mean().item()
    spec = ((total_pixels - tp - fp - fn) / (total_pixels - tp - fn + eps)).mean().item()
    acc = correct_pixels / max(total_pixels, 1)
    bf = float(np.mean(bf_scores))

    logger.info(
        f"Test - Dice: {dice:.4f} | mIoU: {iou:.4f} | Acc: {acc:.4f} | "
        f"Sen: {sen:.4f} | Spe: {spec:.4f} | BF-score: {bf:.4f}"
    )

    return dict(dice=dice, miou=iou, acc=acc, sen=sen, spe=spec, bf=bf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ISIC18')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--crop_size', type=int, nargs='+', default=[224, 224])
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--in_chans', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=96)
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 6, 2])
    parser.add_argument('--decoder_depths', type=int, nargs='+', default=[2, 2, 2, 2])
    parser.add_argument('--num_heads', type=int, nargs='+', default=[3, 6, 12, 24])
    parser.add_argument('--window_size', type=int, default=7)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--run_dir', default='')
    parser.add_argument('--save_samples', type=int, default=10)
    args = parser.parse_args()

    run_dir = args.run_dir or os.path.dirname(os.path.dirname(args.checkpoint))
    log_dir = os.path.join(run_dir, 'log')
    outputs_dir = os.path.join(run_dir, 'outputs')
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))

    dataset = Data(base_dir=args.data_dir, train=False, dataset=args.dataset, crop_szie=args.crop_size)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    metrics = evaluate_test(model, loader, device, args.num_classes, outputs_dir, args.save_samples)

    csv_path = os.path.join(log_dir, 'metrics.csv')
    new_file = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(['phase', 'dice', 'miou', 'acc', 'sen', 'spe', 'bf'])
        w.writerow(['test', metrics['dice'], metrics['miou'],
                    metrics['acc'], metrics['sen'], metrics['spe'], metrics['bf']])


if __name__ == '__main__':
    main()
