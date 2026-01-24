import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from tqdm import tqdm
import setproctitle
import csv
import time
import logging
import cv2

from ZR2ViM import ZR2ViM_Seg

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
    h, w = mask_hw.shape
    if num_classes == 2:
        gray = (mask_hw.astype(np.uint8) * 255)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    palette = [
        (0, 0, 0), (255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0)
    ]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        vis[mask_hw == c] = palette[c % len(palette)]
    return vis


def save_triptych(img_chw: torch.Tensor, gt_hw: torch.Tensor, pred_hw: torch.Tensor,
                  num_classes: int, save_path: str, title_text: str = ""):
    img_vis = _denorm_image(img_chw)
    gt_vis = _colorize_mask(gt_hw.detach().cpu().numpy().astype(np.int64), num_classes)
    pred_vis = _colorize_mask(pred_hw.detach().cpu().numpy().astype(np.int64), num_classes)

    trip = np.concatenate([img_vis, gt_vis, pred_vis], axis=1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, trip)


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            temp_prob = torch.unsqueeze(temp_prob, 1)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes


class CrossEntropyLoss(nn.Module):
    def __init__(self, weights=None, ignore_index=255):
        super(CrossEntropyLoss, self).__init__()
        if weights is not None:
            weights = torch.from_numpy(np.array(weights)).float().cuda()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weights)

    def forward(self, prediction, label):
        loss = self.ce_loss(prediction, label)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, num_classes):
        super(CombinedLoss, self).__init__()
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)

    def forward(self, outputs, targets):
        return self.ce_loss(outputs, targets) + self.dice_loss(outputs, targets)


def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, warmup_iter=None, power=0.9):
    if warmup_iter is not None and cur_iters < warmup_iter:
        lr = base_lr * cur_iters / (warmup_iter + 1e-8)
    elif warmup_iter is not None:
        lr = base_lr * ((1 - float(cur_iters - warmup_iter) / (max_iters - warmup_iter)) ** (power))
    else:
        lr = base_lr * ((1 - float(cur_iters / max_iters)) ** (power))
    optimizer.param_groups[0]['lr'] = lr


def create_model(args):
    model = ZR2ViM_Seg(
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
    return model


def evaluate(model, val_loader, device, num_classes, save_dir=None, epoch=None, vis_samples=2):
    logger = logging.getLogger("train")
    model.eval()
    dice_scores = []

    eps = 1e-7
    tp = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fp = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fn = torch.zeros(num_classes, dtype=torch.float64, device=device)
    total_pixels = 0
    correct_pixels = 0

    saved_vis = 0
    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(val_loader, desc="Validating")):
            images, masks = batch['image'], batch['label']
            names = batch.get('name', None)
            images = images.float()
            images, masks = images.to(device), masks.to(device)
            masks = masks.long().squeeze(1)

            outputs = model(images)

            dice_loss = DiceLoss(num_classes)
            dice_score = 1 - dice_loss(outputs, masks).item()
            dice_scores.append(dice_score)

            preds = torch.argmax(outputs, dim=1)
            total_pixels += masks.numel()
            correct_pixels += (preds == masks).sum().item()
            for c in range(num_classes):
                tp[c] += ((preds == c) & (masks == c)).sum()
                fp[c] += ((preds == c) & (masks != c)).sum()
                fn[c] += ((preds != c) & (masks == c)).sum()

            if (save_dir is not None) and (epoch is not None) and (saved_vis < vis_samples):
                bs = images.size(0)
                save_n = min(bs, vis_samples - saved_vis)
                for j in range(save_n):
                    name_j = None
                    if names is not None:
                        try:
                            name_j = names[j]
                        except Exception:
                            name_j = None
                    stem = os.path.splitext(name_j)[0] if name_j else f"val_{bidx}_{j}"
                    out_path = os.path.join(save_dir, f"{stem}_val_e{epoch}.png")
                    save_triptych(images[j], masks[j], preds[j], num_classes, out_path, title_text=f"{stem} | val e{epoch}")
                saved_vis += save_n

    avg_dice = sum(dice_scores) / max(len(dice_scores), 1)
    total_pixels_t = torch.tensor(float(total_pixels), device=device, dtype=torch.float64)
    iou_per_class = tp / (tp + fp + fn + eps)
    miou = iou_per_class.mean().item()
    sen_per_class = tp / (tp + fn + eps)
    sen = sen_per_class.mean().item()
    spec_per_class = (total_pixels_t - tp - fp - fn) / (total_pixels_t - tp - fn + eps)
    spe = spec_per_class.mean().item()
    acc = correct_pixels / max(total_pixels, 1)

    logger.info(f"Validation - Dice: {avg_dice:.4f} | mIoU: {miou:.4f} | Acc: {acc:.4f} | Sen: {sen:.4f} | Spe: {spe:.4f}")
    return {"dice": avg_dice, "miou": miou, "acc": acc, "sen": sen, "spe": spe}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time_str = time.strftime("%Y%m%d_%H%M%S")
    dataset_name = args.dataset if args.dataset else "dataset"
    run_name = f"{dataset_name}_{start_time_str}_{args.epochs}"
    run_dir = os.path.join(args.output_dir, run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    log_dir = os.path.join(run_dir, 'log')
    outputs_dir = os.path.join(run_dir, 'outputs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(os.path.join(log_dir, 'train.info.log'), mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"运行目录: {run_dir}")
    logger.info(f"使用设备: {device}")

    args.output_dir = run_dir

    model = create_model(args)
    model = model.to(device)

    from dataset import Data
    data_train = Data(train=True, dataset=args.dataset, crop_szie=args.crop_size)
    train_loader = DataLoader(
        data_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    data_val = Data(train=False, dataset=args.dataset, crop_szie=args.crop_size)
    val_loader = DataLoader(
        data_val,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=max(args.num_workers // 2, 0),
        pin_memory=True
    )

    criterion = CombinedLoss(args.num_classes)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )

    metrics_csv_path = os.path.join(log_dir, 'metrics.csv')
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['phase', 'epoch', 'step', 'loss', 'dice', 'miou', 'acc', 'sen', 'spe', 'lr'])

    best_dice = 0.0
    total_iterations = args.epochs * len(train_loader)

    for epoch in range(args.epochs):
        model.train()
        setproctitle.setproctitle(f"ZR2ViM: {epoch}/{args.epochs}")

        running_loss = 0.0
        eps = 1e-7
        acc_tp = torch.zeros(args.num_classes, dtype=torch.float64, device=device)
        acc_fp = torch.zeros(args.num_classes, dtype=torch.float64, device=device)
        acc_fn = torch.zeros(args.num_classes, dtype=torch.float64, device=device)
        acc_total = 0
        acc_correct = 0

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            images, masks = batch['image'], batch['label']
            names = batch.get('name', None)
            images = images.float()
            images, masks = images.to(device), masks.to(device)
            masks = masks.long().squeeze(1)

            current_iter = epoch * len(train_loader) + i
            adjust_learning_rate(
                optimizer,
                args.lr,
                total_iterations,
                current_iter,
                args.warmup_epochs * len(train_loader)
            )
            current_lr = optimizer.param_groups[0]['lr']

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                acc_total += masks.numel()
                acc_correct += (preds == masks).sum().item()
                for c in range(args.num_classes):
                    acc_tp[c] += ((preds == c) & (masks == c)).sum()
                    acc_fp[c] += ((preds == c) & (masks != c)).sum()
                    acc_fn[c] += ((preds != c) & (masks == c)).sum()

                if (i % args.train_save_freq == 0):
                    bs = images.size(0)
                    save_n = min(bs, args.train_save_samples)
                    for j in range(save_n):
                        name_j = None
                        if names is not None:
                            try:
                                name_j = names[j]
                            except Exception:
                                name_j = None
                        stem = os.path.splitext(name_j)[0] if name_j else f"train_{epoch}_{i}_{j}"
                        out_path = os.path.join(outputs_dir, f"{stem}_train_e{epoch+1}_s{i+1}.png")
                        save_triptych(images[j], masks[j], preds[j], args.num_classes, out_path, title_text=f"{stem} | train e{epoch+1} s{i+1}")

            if (i + 1) % args.print_freq == 0:
                acc_total_t = torch.tensor(float(acc_total), device=device, dtype=torch.float64)
                iou_c = acc_tp / (acc_tp + acc_fp + acc_fn + eps)
                miou = iou_c.mean().item()
                sen_c = acc_tp / (acc_tp + acc_fn + eps)
                sen = sen_c.mean().item()
                spe_c = (acc_total_t - acc_tp - acc_fp - acc_fn) / (acc_total_t - acc_tp - acc_fn + eps)
                spe = spe_c.mean().item()
                acc_val = acc_correct / max(acc_total, 1)
                loss_avg = running_loss / args.print_freq
                dice_c = (2 * acc_tp) / (2 * acc_tp + acc_fp + acc_fn + eps)
                dice = dice_c.mean().item()
                logger.info(
                    f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], "
                    f"Loss: {loss_avg:.4f} | Acc: {acc_val:.4f} | mIoU: {miou:.4f} | Sen: {sen:.4f} | Spe: {spe:.4f} | LR: {current_lr:.6f}"
                )
                with open(metrics_csv_path, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['train', epoch + 1, i + 1, float(loss_avg), float(dice), float(miou), float(acc_val), float(sen), float(spe), float(current_lr)])
                running_loss = 0.0
                acc_tp.zero_()
                acc_fp.zero_()
                acc_fn.zero_()
                acc_total = 0
                acc_correct = 0

        if (epoch + 1) % args.eval_freq == 0:
            eval_metrics = evaluate(model, val_loader, device, args.num_classes, save_dir=outputs_dir, epoch=epoch+1)
            with open(metrics_csv_path, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow(['val', epoch + 1, '', '', float(eval_metrics['dice']), float(eval_metrics['miou']), float(eval_metrics['acc']), float(eval_metrics['sen']), float(eval_metrics['spe']), ''])

            if eval_metrics['dice'] > best_dice:
                best_dice = eval_metrics['dice']
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                }, os.path.join(ckpt_dir, 'best_model.pth'))
                logger.info(f"保存最佳模型，Dice分数: {best_dice:.4f}")

    logger.info(f"训练完成！最佳Dice分数: {best_dice:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description='训练ZR2ViM分割模型')

    parser.add_argument('--dataset', type=str, default='', help='数据集名称')
    parser.add_argument('--crop_size', type=int, nargs='+', default=[224, 224], help='裁剪大小 [H, W]')

    parser.add_argument('--img_size', type=int, default=224, help='输入图像大小')
    parser.add_argument('--patch_size', type=int, default=4, help='patch大小')
    parser.add_argument('--in_chans', type=int, default=3, help='输入通道数')
    parser.add_argument('--num_classes', type=int, default=2, help='类别数')
    parser.add_argument('--embed_dim', type=int, default=96, help='嵌入维度')
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 6, 2], help='编码器深度')
    parser.add_argument('--decoder_depths', type=int, nargs='+', default=[2, 2, 2, 2], help='解码器深度')
    parser.add_argument('--num_heads', type=int, nargs='+', default=[3, 6, 12, 24], help='注意力头数')
    parser.add_argument('--window_size', type=int, default=7, help='窗口大小')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='drop path率')

    parser.add_argument('--batch_size', type=int, default=8, help='训练批次大小')
    parser.add_argument('--val_batch_size', type=int, default=4, help='验证批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='预热轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--print_freq', type=int, default=10, help='打印频率')
    parser.add_argument('--eval_freq', type=int, default=1, help='评估频率')

    parser.add_argument('--train_save_freq', type=int, default=100, help='训练阶段每隔多少个step保存一批可视化')
    parser.add_argument('--train_save_samples', type=int, default=1, help='每次保存的训练样本数')
    parser.add_argument('--val_save_samples', type=int, default=2, help='每次验证保存的样本数')

    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)
