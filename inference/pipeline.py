"""
3-class 추론 파이프라인 (normal / ischemic / hemorrhagic).

  1. 분류기 (3-class EfficientNet)  → class_idx ∈ {0,1,2}, softmax 확률 3개
  2. 세그멘터 (3-class U-Net)        → 픽셀별 클래스 맵 (bg/ischemic/hemorrhagic)
  3. 시각화: ischemic=파란톤, hemorrhagic=빨간톤

A 규칙 유지: 분류기 결과 그대로 신뢰. 세그멘터 마스크는 위치 시각화 용도.
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
from models.segmentor import StrokeSegmentor, SEG_CLASS_NAMES


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLS_CLASS_NAMES_DEFAULT = ["normal", "ischemic", "hemorrhagic"]


@dataclass
class PipelineResult:
    class_idx: int
    class_name: str
    confidence: float
    class_probs: dict = field(default_factory=dict)

    # 세그 결과 (3-class)
    lesion_class_map: Optional[np.ndarray] = None   # H×W int64, 값 ∈ {0,1,2}
    ischemic_mask: Optional[np.ndarray] = None      # H×W binary float32
    hemorrhagic_mask: Optional[np.ndarray] = None   # H×W binary float32
    ischemic_area_px: int = 0
    hemorrhagic_area_px: int = 0
    # area_pct 분모는 "뇌 영역 픽셀 수" (brain_area_px). 뇌 마스크 추출 실패 시 H×W 로 fallback.
    ischemic_area_pct: float = 0.0
    hemorrhagic_area_pct: float = 0.0

    # 뇌 영역 (배경·두개골 제외한 뇌조직 추정 마스크)
    brain_mask: Optional[np.ndarray] = None         # H×W uint8 binary
    brain_area_px: int = 0
    # 뇌 영역 내 구성 비율 (normal = 병변 아닌 뇌조직)
    normal_brain_pct: float = 0.0

    overlay_image: Optional[np.ndarray] = None

    def __str__(self):
        lines = [
            f"분류 결과 : {self.class_name.upper()} (신뢰도 {self.confidence:.1%})",
            "클래스 확률: " + " | ".join(
                f"{k}={v:.3f}" for k, v in self.class_probs.items()
            ),
            f"뇌 영역    : {self.brain_area_px}px",
        ]
        if self.ischemic_area_px:
            lines.append(f"ischemic 면적   : {self.ischemic_area_px}px ({self.ischemic_area_pct:.1f}% of brain)")
        if self.hemorrhagic_area_px:
            lines.append(f"hemorrhagic 면적: {self.hemorrhagic_area_px}px ({self.hemorrhagic_area_pct:.1f}% of brain)")
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
                 segmentor_ckpt: Optional[str] = None,
                 cls_image_size: int = 224,
                 seg_image_size: int = 256,
                 device: Optional[torch.device] = None):

        self.device = device or _get_device()
        self.cls_size = cls_image_size
        self.seg_size = seg_image_size

        self.classifier = self._load_classifier(classifier_ckpt)
        self.segmentor = self._load_segmentor(segmentor_ckpt) if segmentor_ckpt else None

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
        self.class_names = ckpt.get("class_names",
                                    cfg.get("class_names", CLS_CLASS_NAMES_DEFAULT))
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
            num_classes=ckpt.get("num_classes", 3),
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device).eval()
        self.seg_class_names = ckpt.get("class_names", SEG_CLASS_NAMES)
        return model

    @torch.no_grad()
    def run(self, image_input) -> PipelineResult:
        orig_np = self._load_image(image_input)
        h, w = orig_np.shape[:2]

        # 분류
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

        # 뇌 영역 마스크 (분모 용도) — 세그 유무와 무관하게 항상 계산
        brain_mask = _compute_brain_mask(orig_np)
        brain_area = int(brain_mask.sum())
        denom = brain_area if brain_area > 0 else (h * w)
        result.brain_mask = brain_mask
        result.brain_area_px = brain_area

        # 세그 (있을 때만)
        if self.segmentor is not None:
            seg_tensor = self.seg_transform(image=orig_np)["image"].unsqueeze(0).to(self.device)
            cls_map = self.segmentor.predict_mask(seg_tensor)[0].cpu().numpy().astype(np.int64)
            cls_map_full = self._resize_class_map(cls_map, h, w)

            # 세그 결과를 뇌 영역 내로 제한 (배경·두개골 위의 잘못된 예측 제거)
            if brain_area > 0:
                cls_map_full = cls_map_full * brain_mask

            isc_bin = (cls_map_full == 1).astype(np.float32)
            hem_bin = (cls_map_full == 2).astype(np.float32)

            result.lesion_class_map = cls_map_full
            if isc_bin.sum() > 0:
                result.ischemic_mask = isc_bin
                result.ischemic_area_px = int(isc_bin.sum())
                result.ischemic_area_pct = result.ischemic_area_px / denom * 100
            if hem_bin.sum() > 0:
                result.hemorrhagic_mask = hem_bin
                result.hemorrhagic_area_px = int(hem_bin.sum())
                result.hemorrhagic_area_pct = result.hemorrhagic_area_px / denom * 100

        result.normal_brain_pct = max(
            0.0, 100.0 - result.ischemic_area_pct - result.hemorrhagic_area_pct
        )
        result.overlay_image = self._overlay(orig_np, result)
        return result

    def _load_image(self, inp) -> np.ndarray:
        if isinstance(inp, np.ndarray):
            if inp.ndim == 2:
                return np.stack([inp] * 3, axis=-1)
            return inp
        img = Image.open(inp).convert("RGB") if not isinstance(inp, Image.Image) else inp.convert("RGB")
        return np.array(img)

    @staticmethod
    def _resize_class_map(cls_map: np.ndarray, h: int, w: int) -> np.ndarray:
        import cv2
        return cv2.resize(cls_map.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(np.int64)

    @staticmethod
    def _overlay(orig: np.ndarray, r: PipelineResult,
                 alpha: float = 0.45) -> np.ndarray:
        out = orig.copy()
        if r.ischemic_mask is not None:
            color = np.array([60, 120, 255], dtype=np.uint8)  # 파란톤
            m = r.ischemic_mask > 0
            out[m] = ((1 - alpha) * out[m] + alpha * color).astype(np.uint8)
        if r.hemorrhagic_mask is not None:
            color = np.array([255, 80, 80], dtype=np.uint8)   # 빨간톤
            m = r.hemorrhagic_mask > 0
            out[m] = ((1 - alpha) * out[m] + alpha * color).astype(np.uint8)
        return out


def _compute_brain_mask(orig_np: np.ndarray) -> np.ndarray:
    """CT 이미지에서 뇌 영역(배경·두개골 외곽 제외) 근사 마스크를 추정.
    threshold → morphological close → 외부 flood fill 로 홀 채움 → 최대 연결성분.
    반환: H×W uint8 binary (1 = 뇌조직 후보).
    """
    import cv2
    if orig_np.ndim == 3:
        gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = orig_np.astype(np.uint8)

    _, bw = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    # 외부 배경 flood fill 로 홀(두개골 안쪽 어두운 영역) 채우기
    h, w = bw.shape
    flood = bw.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    filled = cv2.bitwise_or(bw, cv2.bitwise_not(flood))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
    if n <= 1:
        return (filled > 0).astype(np.uint8)
    # background(=0) 제외한 최대 연결성분을 뇌로 간주
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)
