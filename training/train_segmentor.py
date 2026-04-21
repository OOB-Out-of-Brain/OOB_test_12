"""
AISD 데이터셋으로 U-Net 분할 모델 학습.

실행:
    python training/train_segmentor.py
    python training/train_segmentor.py --epochs 50 --batch_size 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import yaml

from data.segmentation_dataset import build_segmentation_dataloaders
from models.segmentor import StrokeSegmentor
from training.metrics import DiceBCELoss, dice_score, iou_score


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_dice = 0.0, 0.0

    for images, masks in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        pred_masks = (torch.sigmoid(logits) > 0.5).float()
        total_loss += loss.item()
        total_dice += dice_score(pred_masks, masks)

    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0

    for images, masks in tqdm(loader, desc="  eval ", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        pred_masks = (torch.sigmoid(logits) > 0.5).float()

        total_loss += loss.item()
        total_dice += dice_score(pred_masks, masks)
        total_iou += iou_score(pred_masks, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def main(args):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    s = cfg["segmentor"]
    d = cfg["data"]

    epochs = args.epochs or s["epochs"]
    batch_size = args.batch_size or s["batch_size"]
    lr = args.lr or s["learning_rate"]
    image_size = s["image_size"]
    save_path = Path(s["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"\n디바이스: {device}")
    print(f"설정: epochs={epochs}, batch={batch_size}, lr={lr}, img={image_size}\n")

    aisd_path = d["aisd_path"]
    print(f"AISD 데이터셋 로딩: {aisd_path}")
    train_loader, val_loader = build_segmentation_dataloaders(
        aisd_root=aisd_path,
        image_size=image_size,
        batch_size=batch_size,
    )
    print(f"학습: {len(train_loader.dataset)}개  검증: {len(val_loader.dataset)}개\n")

    model = StrokeSegmentor(
        encoder_name=s["encoder"],
        encoder_weights=s["encoder_weights"],
    ).to(device)

    criterion = DiceBCELoss(
        dice_weight=s["dice_weight"],
        bce_weight=s["bce_weight"],
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=s["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_dice = 0.0
    patience_counter = 0
    patience = s["early_stopping_patience"]

    for epoch in range(1, epochs + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train loss={train_loss:.4f} dice={train_dice:.4f} | "
              f"Val loss={val_loss:.4f} dice={val_dice:.4f} iou={val_iou:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
                "config": s,
            }, save_path / "best_segmentor.pth")
            print(f"  → 모델 저장 (best Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping (patience={patience})")
                break

    print(f"\n학습 완료. 저장 경로: {save_path / 'best_segmentor.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()
    main(args)
