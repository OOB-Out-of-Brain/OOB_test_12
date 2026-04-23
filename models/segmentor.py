"""
3-class 뇌졸중 병변 분할 모델 (background / ischemic / hemorrhagic).

출력: (B, 3, H, W) — 픽셀별 softmax logit
  채널 0 = background (정상 뇌/두개골/배경)
  채널 1 = ischemic  (허혈)
  채널 2 = hemorrhagic (출혈)
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


SEG_CLASS_NAMES = ["background", "ischemic", "hemorrhagic"]
SEG_NUM_CLASSES = 3


class StrokeSegmentor(nn.Module):
    def __init__(self, encoder_name: str = "resnet34",
                 encoder_weights: str = "imagenet",
                 num_classes: int = SEG_NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)

    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """픽셀별 argmax 클래스 맵 (B, H, W) — 값 ∈ {0,1,2}"""
        logits = self.forward(x)
        return logits.argmax(dim=1)

    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        """픽셀별 softmax 확률 (B, num_classes, H, W)"""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
