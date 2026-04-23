"""
3-class U-Net 분할 학습 (background / ischemic / hemorrhagic).

데이터:
  - CT Hemorrhage (정상 + 출혈 마스크)
  - BHSD (출혈 마스크)
  - AISD (허혈 마스크)

실행:
    python training/train_segmentor_3class.py
    python training/train_segmentor_3class.py --epochs 50 --batch_size 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import yaml

from data.seg_3class_dataset import build_seg_3class_dataloaders
from models.segmentor_3class import StrokeSegmentor3Class, SEG_CLASS_NAMES, SEG_NUM_CLASSES


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MultiDiceCELoss(nn.Module):
    """다중 클래스 Dice + CrossEntropy 조합.
    logits: (B, C, H, W), target: (B, H, W) long
    """

    def __init__(self, num_classes: int, dice_weight: float = 0.6,
                 ce_weight: float = 0.4, class_weights=None, ignore_bg_in_dice: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.dice_w = dice_weight
        self.ce_w = ce_weight
        self.ignore_bg_in_dice = ignore_bg_in_dice
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)

        probs = F.softmax(logits, dim=1)
        target_1h = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        start = 1 if self.ignore_bg_in_dice else 0

        dims = (0, 2, 3)
        intersection = (probs[:, start:] * target_1h[:, start:]).sum(dim=dims)
        card = probs[:, start:].sum(dim=dims) + target_1h[:, start:].sum(dim=dims)
        dice = (2 * intersection + 1) / (card + 1)
        dice_loss = 1 - dice.mean()

        return self.dice_w * dice_loss + self.ce_w * ce_loss


def per_class_dice(pred_mask, target, num_classes: int):
    """pred_mask, target: (B, H, W) long. 반환: length=num_classes 리스트 (각 클래스 Dice)."""
    dices = []
    for c in range(num_classes):
        p = (pred_mask == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        card = p.sum() + t.sum()
        if card.item() == 0:
            dices.append(float("nan"))
        else:
            dices.append(((2 * inter + 1) / (card + 1)).item())
    return dices


def train_one_epoch(model, loader, criterion, optimizer, device, num_classes):
    model.train()
    total_loss = 0.0
    dice_sum = np.zeros(num_classes)
    dice_cnt = np.zeros(num_classes)

    for images, masks in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            d = per_class_dice(preds, masks, num_classes)
            for i, v in enumerate(d):
                if not np.isnan(v):
                    dice_sum[i] += v
                    dice_cnt[i] += 1
        total_loss += loss.item()

    n = len(loader)
    mean_dice = np.where(dice_cnt > 0, dice_sum / np.maximum(dice_cnt, 1), float("nan"))
    return total_loss / n, mean_dice


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    dice_sum = np.zeros(num_classes)
    dice_cnt = np.zeros(num_classes)

    for images, masks in tqdm(loader, desc="  eval ", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        preds = logits.argmax(dim=1)

        total_loss += loss.item()
        d = per_class_dice(preds, masks, num_classes)
        for i, v in enumerate(d):
            if not np.isnan(v):
                dice_sum[i] += v
                dice_cnt[i] += 1

    n = len(loader)
    mean_dice = np.where(dice_cnt > 0, dice_sum / np.maximum(dice_cnt, 1), float("nan"))
    return total_loss / n, mean_dice


def main(args):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    s = cfg["segmentor"]
    d = cfg["data"]

    epochs = args.epochs or s["epochs"]
    batch_size = args.batch_size or s["batch_size"]
    lr = args.lr or s["learning_rate"]
    image_size = s["image_size"]
    save_path = Path(args.save_path or "./checkpoints/segmentor_3class")
    save_path.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"\n디바이스: {device}")
    print(f"설정: epochs={epochs}, batch={batch_size}, lr={lr}, img={image_size}")
    print(f"세그 클래스({SEG_NUM_CLASSES}): {SEG_CLASS_NAMES}\n")

    ct_path = d["ct_hemorrhage_path"]
    aisd_path = d["aisd_path"]
    print("데이터셋 로딩 (CT Hemorrhage + BHSD + AISD)")
    train_loader, val_loader = build_seg_3class_dataloaders(
        ct_root=ct_path,
        aisd_root=aisd_path,
        bhsd_processed_dir="./data/processed/bhsd",
        image_size=image_size,
        batch_size=batch_size,
    )
    print(f"학습: {len(train_loader.dataset)}개  검증: {len(val_loader.dataset)}개\n")

    model = StrokeSegmentor3Class(
        encoder_name=s["encoder"],
        encoder_weights=s["encoder_weights"],
        num_classes=SEG_NUM_CLASSES,
    ).to(device)

    criterion = MultiDiceCELoss(
        num_classes=SEG_NUM_CLASSES,
        dice_weight=s["dice_weight"],
        ce_weight=s["bce_weight"],  # 다중분류이므로 CE weight로 재사용
        ignore_bg_in_dice=True,     # background 때문에 Dice 쏠림 방지
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=s["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_score = 0.0
    patience_counter = 0
    patience = s["early_stopping_patience"]

    def _fmt(dices):
        return "  ".join(f"{n}={v:.3f}" for n, v in zip(SEG_CLASS_NAMES, dices))

    for epoch in range(1, epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, SEG_NUM_CLASSES
        )
        val_loss, val_dice = evaluate(
            model, val_loader, criterion, device, SEG_NUM_CLASSES
        )
        scheduler.step()

        # lesion-only 평균 (background 제외) 로 best 판정
        lesion_mean = float(np.nanmean(val_dice[1:]))

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train loss={train_loss:.4f}  {_fmt(train_dice)} | "
              f"val loss={val_loss:.4f}  {_fmt(val_dice)} | lesion_dice={lesion_mean:.4f}")

        if lesion_mean > best_score:
            best_score = lesion_mean
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_dice_per_class": val_dice.tolist(),
                "val_lesion_dice": lesion_mean,
                "class_names": SEG_CLASS_NAMES,
                "num_classes": SEG_NUM_CLASSES,
                "config": s,
            }, save_path / "best_segmentor_3class.pth")
            print(f"  → 모델 저장 (best lesion Dice: {best_score:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping (patience={patience})")
                break

    print(f"\n학습 완료. 저장 경로: {save_path / 'best_segmentor_3class.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save_path", type=str, default=None,
                        help="기본: ./checkpoints/segmentor_3class")
    args = parser.parse_args()
    main(args)
