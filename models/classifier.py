import torch
import torch.nn as nn
import timm


class StrokeClassifier(nn.Module):
    """
    EfficientNet-B2 기반 3분류 모델.
    출력: normal(0) / ischemic(1) / hemorrhagic(2)
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def predict(self, x: torch.Tensor):
        """softmax 확률과 class index 반환"""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        return pred, probs
