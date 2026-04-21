import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# ── Classification ────────────────────────────────────────────────────────────

def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (preds == targets).float().mean().item()


def cls_report(preds: np.ndarray, targets: np.ndarray,
               class_names: list) -> str:
    return classification_report(targets, preds, target_names=class_names, digits=4)


def conf_matrix(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return confusion_matrix(targets, preds)


# ── Segmentation ──────────────────────────────────────────────────────────────

def dice_score(pred: torch.Tensor, target: torch.Tensor,
               smooth: float = 1e-6) -> float:
    """배치 평균 Dice."""
    pred = pred.float().view(pred.size(0), -1)
    target = target.float().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return dice.mean().item()


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1e-6) -> float:
    """배치 평균 IoU."""
    pred = pred.float().view(pred.size(0), -1)
    target = target.float().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return ((intersection + smooth) / (union + smooth)).mean().item()


# ── Loss functions ────────────────────────────────────────────────────────────

class DiceBCELoss(torch.nn.Module):
    """Dice + BCE 조합 손실. 의료 영상 분할에 적합."""

    def __init__(self, dice_weight: float = 0.6, bce_weight: float = 0.4):
        super().__init__()
        self.dice_w = dice_weight
        self.bce_w = bce_weight
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        prob = torch.sigmoid(logits)
        prob_flat = prob.view(prob.size(0), -1)
        tgt_flat = targets.view(targets.size(0), -1)
        intersection = (prob_flat * tgt_flat).sum(dim=1)
        dice_loss = 1 - (2 * intersection + 1) / (prob_flat.sum(dim=1) + tgt_flat.sum(dim=1) + 1)
        dice_loss = dice_loss.mean()

        return self.dice_w * dice_loss + self.bce_w * bce_loss
