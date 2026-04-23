"""
3-class 분류 학습 스크립트 (normal / ischemic / hemorrhagic).

tekno21 라벨 체계를 기준으로 CT Hemorrhage, BHSD도 동일 3-class 공간으로 매핑하여 학습.
  - normal=0, ischemic=1, hemorrhagic=2
  - tekno21 iskemi(허혈) 샘플을 제거하지 않고 전부 사용

실행:
    python training/train_classifier_3class.py
    python training/train_classifier_3class.py --epochs 30 --batch_size 8
    python training/train_classifier_3class.py --tekno21-only    # CT/BHSD 제외, tekno21만 학습
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

from data.combined_dataset_3class import (
    build_combined_3class_dataloaders,
    CLASS_NAMES,
    NUM_CLASSES,
)
from models.classifier import StrokeClassifier
from training.metrics import accuracy, cls_report, conf_matrix


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item()
        total_acc += accuracy(preds, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item()
        total_acc += accuracy(preds, labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(loader)
    return (total_loss / n, total_acc / n,
            np.array(all_preds), np.array(all_labels))


def main(args):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    c = cfg["classifier"]

    epochs = args.epochs or c["epochs"]
    batch_size = args.batch_size or c["batch_size"]
    lr = args.lr or c["learning_rate"]
    image_size = c["image_size"]
    save_path = Path(args.save_path or "./checkpoints/classifier_3class")
    save_path.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"\n디바이스: {device}")
    print(f"설정: epochs={epochs}, batch={batch_size}, lr={lr}, img={image_size}")
    print(f"클래스({NUM_CLASSES}): {CLASS_NAMES}\n")

    ct_path = cfg["data"]["ct_hemorrhage_path"]
    tk_cache = cfg["data"]["tekno21_cache"]
    print("데이터셋 로딩 (3-class: tekno21 + CT Hemorrhage + BHSD)")
    train_loader, val_loader, class_weights = build_combined_3class_dataloaders(
        ct_root=ct_path, tekno21_cache=tk_cache,
        image_size=image_size, batch_size=batch_size,
        use_ct=not args.tekno21_only,
        use_bhsd=not args.tekno21_only,
    )
    print(f"학습: {len(train_loader.dataset)}개  검증: {len(val_loader.dataset)}개")
    print(f"class_weights = {class_weights.tolist()}\n")

    model = StrokeClassifier(
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout_rate=c["dropout_rate"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=c["label_smoothing"],
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=c["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_acc = 0.0
    patience_counter = 0
    patience = c["early_stopping_patience"]

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Val loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "class_names": CLASS_NAMES,
                "num_classes": NUM_CLASSES,
                "config": c,
            }, save_path / "best_classifier_3class.pth")
            print(f"  → 모델 저장 (best val acc: {best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping (patience={patience})")
                break

    print(f"\n최종 검증 리포트 (best model):")
    print(cls_report(val_preds, val_labels, CLASS_NAMES))
    print("Confusion matrix (rows=true, cols=pred):")
    print("           " + "  ".join(f"{n:>11}" for n in CLASS_NAMES))
    cm = conf_matrix(val_preds, val_labels)
    for name, row in zip(CLASS_NAMES, cm):
        print(f"  {name:>9} " + "  ".join(f"{v:>11d}" for v in row))
    print(f"\n학습 완료. 저장 경로: {save_path / 'best_classifier_3class.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save_path", type=str, default=None,
                        help="기본: ./checkpoints/classifier_3class")
    parser.add_argument("--tekno21-only", action="store_true",
                        help="CT Hemorrhage / BHSD 제외하고 tekno21만으로 학습")
    args = parser.parse_args()
    main(args)
