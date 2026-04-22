"""
3단계 추론 파이프라인:
  1. 분류 (Normal / Ischemic / Hemorrhagic)
  2. 허혈성인 경우 병변 분할 (U-Net)
  3. 시각화 결과 반환
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.classifier import StrokeClassifier
from models.segmentor import StrokeSegmentor


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class PipelineResult:
    # 분류 결과
    class_idx: int
    class_name: str
    confidence: float
    class_probs: dict = field(default_factory=dict)

    # 분할 결과 (허혈성인 경우에만)
    lesion_mask: Optional[np.ndarray] = None   # H×W binary float32
    lesion_area_px: int = 0
    lesion_area_pct: float = 0.0

    # 시각화
    overlay_image: Optional[np.ndarray] = None  # H×W×3 uint8

    def __str__(self):
        lines = [
            f"분류 결과  : {self.class_name.upper()} (신뢰도 {self.confidence:.1%})",
            f"클래스 확률: " + " | ".join(
                f"{k}={v:.3f}" for k, v in self.class_probs.items()
            ),
        ]
        if self.lesion_mask is not None:
            lines.append(f"병변 면적  : {self.lesion_area_px}px ({self.lesion_area_pct:.1f}%)")
        return "\n".join(lines)


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class StrokePipeline:
    def __init__(self,
                 classifier_ckpt: str,
                 segmentor_ckpt: str,
                 cls_image_size: int = 224,
                 seg_image_size: int = 256,
                 seg_threshold: float = 0.5,
                 device: Optional[torch.device] = None):

        self.device = device or _get_device()
        self.cls_size = cls_image_size
        self.seg_size = seg_image_size
        self.seg_threshold = seg_threshold

        self.classifier = self._load_classifier(classifier_ckpt)
        self.segmentor = self._load_segmentor(segmentor_ckpt)

        self.cls_transform = A.Compose([
            A.Resize(cls_image_size, cls_image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        self.seg_transform = A.Compose([
            A.Resize(seg_image_size, seg_image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def _load_classifier(self, ckpt_path: str) -> StrokeClassifier:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        cfg = ckpt.get("config", {})
        self.class_names = ckpt.get("class_names", cfg.get("class_names", ["normal", "hemorrhagic"]))
        model = StrokeClassifier(
            num_classes=len(self.class_names),
            pretrained=False,
            dropout_rate=cfg.get("dropout_rate", 0.3),
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device).eval()
        return model

    def _load_segmentor(self, ckpt_path: str) -> StrokeSegmentor:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        cfg = ckpt.get("config", {})
        model = StrokeSegmentor(
            encoder_name=cfg.get("encoder", "resnet34"),
            encoder_weights=None,
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device).eval()
        return model

    @torch.no_grad()
    def run(self, image_input) -> PipelineResult:
        """
        image_input: 파일 경로(str/Path), PIL.Image, 또는 np.ndarray (H×W×3 uint8)
        """
        orig_np = self._load_image(image_input)

        # ── 1단계: 분류 ──────────────────────────────────────────────────────
        cls_tensor = self.cls_transform(image=orig_np)["image"].unsqueeze(0).to(self.device)
        pred_idx, probs = self.classifier.predict(cls_tensor)

        pred_idx = pred_idx.item()
        probs_np = probs.cpu().numpy()[0]
        class_name = self.class_names[pred_idx]

        result = PipelineResult(
            class_idx=pred_idx,
            class_name=class_name,
            confidence=float(probs_np[pred_idx]),
            class_probs={name: float(p) for name, p in zip(self.class_names, probs_np)},
        )

        # ── 2단계: 세그멘테이션 (항상 실행) ────────────────────────────────
        seg_tensor = self.seg_transform(image=orig_np)["image"].unsqueeze(0).to(self.device)
        mask_tensor = self.segmentor.predict_mask(seg_tensor, self.seg_threshold)
        mask_resized = self._resize_mask(
            mask_tensor[0, 0].cpu().numpy(),
            target_h=orig_np.shape[0],
            target_w=orig_np.shape[1],
        )
        total_px = orig_np.shape[0] * orig_np.shape[1]
        lesion_px = int(mask_resized.sum())
        lesion_pct = lesion_px / total_px * 100

        # 최종 판독 규칙:
        #   병변 비율 ≤ 1% → normal (미세 오탐은 무시)
        #   병변 비율 >  1% & 분류기 normal → hemorrhagic으로 override
        if lesion_pct <= 1.0:
            if "normal" in self.class_names:
                result.class_idx = self.class_names.index("normal")
                result.class_name = "normal"
        elif class_name == "normal":
            if "hemorrhagic" in self.class_names:
                result.class_idx = self.class_names.index("hemorrhagic")
                result.class_name = "hemorrhagic"

        if lesion_px > 0:
            result.lesion_mask = mask_resized
            result.lesion_area_px = lesion_px
            result.lesion_area_pct = lesion_pct

        # ── 3단계: 시각화 ────────────────────────────────────────────────────
        from inference.visualization import visualize_result
        result.overlay_image = visualize_result(orig_np, result)

        return result

    def _load_image(self, inp) -> np.ndarray:
        if isinstance(inp, np.ndarray):
            return inp if inp.ndim == 3 else np.stack([inp] * 3, axis=-1)
        img = Image.open(inp).convert("RGB") if not isinstance(inp, Image.Image) else inp.convert("RGB")
        return np.array(img)

    @staticmethod
    def _resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        import cv2
        resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return (resized > 0.5).astype(np.float32)
