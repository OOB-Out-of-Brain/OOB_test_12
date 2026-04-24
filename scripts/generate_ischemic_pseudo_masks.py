"""Grad-CAM 으로 tekno21 iskemi 이미지에 pseudo-ischemic-mask 생성.

세그멘터 학습용. 세그멘터 원래 허혈 학습이 AISD 합성 400장뿐이라 tekno21 실제
CT에서 허혈 위치를 못 잡음. 분류기는 tekno21 iskemi Val Recall 98.53% 로 잘 맞히니까,
분류기의 Grad-CAM 을 허혈 위치의 proxy 로 써서 pseudo-mask 를 만든다.

알고리즘:
  1. 분류기 (EfficientNet-B2) forward → last conv feature map f
  2. logit[ischemic] 을 f 에 대해 backward → grad g
  3. weights = g.mean(2,3) → (C,1,1)
  4. CAM = ReLU(sum(weights * f, dim=C))
  5. 원본 크기로 upsample, min-max normalize
  6. threshold (top 20% 픽셀) 로 binary mask

출력:
  data/processed/tekno21_isch_pseudo/
    ├─ images/<idx>.png
    ├─ masks/<idx>.png   (0/255 binary)
    └─ index.csv         (image_path, mask_path)
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset

from models.classifier import StrokeClassifier


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ISCHEMIC_LABEL_IDX = 1  # classifier class index
TEKNO21_ISKEMI_LABEL = 1  # tekno21 원본 라벨 (0=Kanama 1=iskemi 2=Inme Yok)
CAM_THRESHOLD_QUANTILE = 0.80  # 상위 20% 를 병변으로
OUT_DIR = Path("./data/processed/tekno21_isch_pseudo")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_classifier(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ck.get("config", {})
    model = StrokeClassifier(
        num_classes=len(ck.get("class_names", ["normal", "ischemic", "hemorrhagic"])),
        pretrained=False,
        dropout_rate=cfg.get("dropout_rate", 0.3),
    )
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    return model


def make_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class GradCAM:
    """Hook 기반 Grad-CAM. EfficientNet-B2 의 마지막 conv feature 에 attach."""

    def __init__(self, model, target_module):
        self.model = model
        self.features = None
        self.grad = None
        self._fh = target_module.register_forward_hook(self._save_features)
        self._bh = target_module.register_full_backward_hook(self._save_grad)

    def _save_features(self, m, inp, out):
        self.features = out

    def _save_grad(self, m, grad_in, grad_out):
        self.grad = grad_out[0]

    def generate(self, logits, class_idx: int) -> torch.Tensor:
        """입력 (1, H, W) CAM 을 얻고 min-max 정규화."""
        self.model.zero_grad()
        target = logits[0, class_idx]
        target.backward(retain_graph=False)

        # weights: GAP of grad over spatial → (1, C, 1, 1)
        weights = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.features).sum(dim=1))  # (1, H', W')
        cam = F.interpolate(cam.unsqueeze(1), size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False).squeeze(1).squeeze(0)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            return torch.zeros_like(cam)
        return (cam - cam_min) / (cam_max - cam_min)

    def close(self):
        self._fh.remove()
        self._bh.remove()


def main():
    device = get_device()
    print(f"device: {device}")

    ck_path = "./checkpoints/classifier/best_classifier.pth"
    if not Path(ck_path).exists():
        sys.exit(f"  분류기 체크포인트 없음: {ck_path}")
    print(f"분류기 로드: {ck_path}")
    model = build_classifier(ck_path, device)

    # EfficientNet-B2 의 마지막 conv: backbone.conv_head (timm)
    # 이게 없으면 backbone.bn2 또는 마지막 conv_block 의 출력을 쓸 수 있음.
    backbone = model.backbone
    target = None
    for name in ["conv_head", "bn2"]:
        if hasattr(backbone, name):
            target = getattr(backbone, name)
            print(f"  Grad-CAM target: backbone.{name}")
            break
    if target is None:
        # fallback: 마지막 conv block 의 out
        target = list(backbone.children())[-3]
        print(f"  Grad-CAM target: 마지막 block (fallback)")

    cam_generator = GradCAM(model, target)

    print("tekno21 로드 + iskemi 인덱스 수집...")
    ds = load_dataset(
        "BTX24/tekno21-brain-stroke-dataset-multi",
        split="train",
        cache_dir="./data/raw/tekno21",
    )
    iskemi_idx = [i for i in range(len(ds)) if int(ds[i]["label"]) == TEKNO21_ISKEMI_LABEL]
    print(f"  iskemi {len(iskemi_idx)} 장")

    img_dir = OUT_DIR / "images"
    msk_dir = OUT_DIR / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    transform = make_transform()

    index_rows = []
    kept = 0
    skipped_wrong_pred = 0

    for i in tqdm(iskemi_idx, desc="CAM"):
        item = ds[i]
        pil = item["image"]
        if not isinstance(pil, Image.Image):
            pil = Image.fromarray(pil)
        orig_np = np.array(pil.convert("RGB"))

        tensor = transform(image=orig_np)["image"].unsqueeze(0).to(device)
        tensor.requires_grad_(False)

        # Forward
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(probs.argmax().item())

        # 분류기 자체가 ischemic 로 예측한 샘플만 pseudo-mask 로 사용 (노이즈 방지)
        if pred != ISCHEMIC_LABEL_IDX:
            skipped_wrong_pred += 1
            continue

        # Grad-CAM
        cam = cam_generator.generate(logits, ISCHEMIC_LABEL_IDX)  # (H, W) 0~1
        cam_np = cam.detach().cpu().numpy()

        # Threshold
        thr = np.quantile(cam_np, CAM_THRESHOLD_QUANTILE)
        mask = (cam_np > thr).astype(np.uint8) * 255

        # Pseudo-mask 가 너무 작거나 크면 스킵 (노이즈)
        frac = (mask > 0).mean()
        if frac < 0.01 or frac > 0.4:
            skipped_wrong_pred += 1
            continue

        # 원본 이미지 & 마스크 모두 224 로 저장 (세그 로더가 다시 resize 하지만 일관성)
        orig_224 = A.Resize(IMAGE_SIZE, IMAGE_SIZE)(image=orig_np)["image"]
        img_path = img_dir / f"{i:05d}.png"
        msk_path = msk_dir / f"{i:05d}.png"
        Image.fromarray(orig_224).save(img_path)
        Image.fromarray(mask).save(msk_path)
        index_rows.append((
            str(img_path.relative_to(OUT_DIR)),
            str(msk_path.relative_to(OUT_DIR)),
            i, float(probs[ISCHEMIC_LABEL_IDX].item()), float(frac),
        ))
        kept += 1

    cam_generator.close()

    # index.csv
    with open(OUT_DIR / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "mask_path", "tekno21_idx", "classifier_conf", "mask_fraction"])
        w.writerows(index_rows)

    print(f"\n저장: {OUT_DIR}")
    print(f"  사용된 pseudo-mask: {kept} 장")
    print(f"  분류기가 ischemic 아니라 판단/마스크 불량으로 skip: {skipped_wrong_pred}")


if __name__ == "__main__":
    main()
