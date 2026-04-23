"""
3-class 세그멘테이션용 데이터셋 (background=0, ischemic=1, hemorrhagic=2).

데이터 소스:
  - CT Hemorrhage : *_HGE_Seg.jpg (binary) → 픽셀값>0 → 2 (hemorrhagic)
  - BHSD         : processed/bhsd 의 binary mask → 2 (hemorrhagic)
  - AISD         : synthetic mask (binary) → 1 (ischemic)

각 PNG 마스크는 binary (0/255)라서 자동으로 해당 클래스 인덱스로 변환된다.
배경은 별도 데이터 없이, 마스크에 해당 lesion이 없는 픽셀 전부가 class 0.
"""

from pathlib import Path
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

BG, ISCHEMIC, HEMORRHAGIC = 0, 1, 2


def _seg_transforms(image_size: int, split: str) -> A.Compose:
    extra = {"mask": "mask"}
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ], additional_targets=extra)
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ], additional_targets=extra)


class Seg3ClassDataset(Dataset):
    """samples: list of (img_path, mask_path_or_None, lesion_class_int)
       mask_path=None 이면 정상 슬라이스 (전부 background)."""

    def __init__(self, samples, image_size, split):
        self.samples = samples
        self.transform = _seg_transforms(image_size, split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, lesion_class = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]

        if mask_path is None:
            mask_idx = np.zeros((h, w), dtype=np.int64)
        else:
            m = np.array(Image.open(mask_path).convert("L"))
            bin_mask = (m > 127).astype(np.int64)
            mask_idx = bin_mask * int(lesion_class)

        out = self.transform(image=image, mask=mask_idx)
        mask_tensor = out["mask"].long()  # (H, W) — 클래스 인덱스
        return out["image"], mask_tensor


def _collect_ct_hemorrhage(data_root: str):
    import pandas as pd
    root = Path(data_root)
    df = pd.read_csv(root / "hemorrhage_diagnosis.csv")
    df.columns = df.columns.str.strip()
    out = []
    for _, row in df.iterrows():
        pid_int = int(row["PatientNumber"])
        pid_str = str(pid_int).zfill(3)
        sn = int(row["SliceNumber"])
        img = root / "Patients_CT" / pid_str / "brain" / f"{sn}.jpg"
        if not img.exists():
            continue
        mask = root / "Patients_CT" / pid_str / "brain" / f"{sn}_HGE_Seg.jpg"
        if int(row["No_Hemorrhage"]) == 1:
            out.append((img, None, BG, f"ct_{pid_int}"))
        else:
            if mask.exists():
                out.append((img, mask, HEMORRHAGIC, f"ct_{pid_int}"))
            else:
                out.append((img, None, BG, f"ct_{pid_int}"))
    return out


def _collect_bhsd_seg(processed_dir: str):
    root = Path(processed_dir)
    idx_csv = root / "index.csv"
    if not idx_csv.exists():
        print(f"  ⚠️ BHSD index 없음: {idx_csv}")
        return []
    out = []
    with open(idx_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = root / row["image_path"]
            msk = root / row["mask_path"]
            if not (img.exists() and msk.exists()):
                continue
            stem = img.stem
            pid = stem.rsplit("_s", 1)[0]
            out.append((img, msk, HEMORRHAGIC, f"bhsd_{pid}"))
    return out


def _collect_aisd(aisd_root: str):
    root = Path(aisd_root)
    img_dir = root / "images"
    msk_dir = root / "masks"
    if not img_dir.exists():
        print(f"  ⚠️ AISD 없음: {img_dir}")
        return []
    out = []
    for img in sorted(img_dir.glob("*.png")):
        msk = msk_dir / img.name
        if not msk.exists():
            continue
        out.append((img, msk, ISCHEMIC, f"aisd_{img.stem}"))
    return out


def _patient_split(samples, val_ratio, seed):
    pids = sorted({s[3] for s in samples})
    rng = np.random.RandomState(seed)
    rng.shuffle(pids)
    n_val = max(1, int(len(pids) * val_ratio))
    val_set = set(pids[:n_val])
    train = [(s[0], s[1], s[2]) for s in samples if s[3] not in val_set]
    val   = [(s[0], s[1], s[2]) for s in samples if s[3] in val_set]
    return train, val


def build_seg_dataloaders(ct_root: str, aisd_root: str,
                                  bhsd_processed_dir: str,
                                  image_size: int, batch_size: int,
                                  val_ratio: float = 0.2, seed: int = 42,
                                  num_workers: int = 2,
                                  include_ct_normal: bool = True):
    """
    Args:
        include_ct_normal: CT Hemorrhage의 정상 슬라이스를 함께 학습 (배경만 있는 pair)
                           False 면 lesion 있는 슬라이스만 사용.
    """
    print("  CT Hemorrhage 세그 로딩...")
    ct_all = _collect_ct_hemorrhage(ct_root)
    if not include_ct_normal:
        ct_all = [s for s in ct_all if s[2] != BG]

    print("  BHSD 세그 로딩...")
    bhsd_all = _collect_bhsd_seg(bhsd_processed_dir)

    print("  AISD (ischemic) 세그 로딩...")
    aisd_all = _collect_aisd(aisd_root)

    ct_tr, ct_va = _patient_split(ct_all, val_ratio, seed)
    bh_tr, bh_va = _patient_split(bhsd_all, val_ratio, seed + 1) if bhsd_all else ([], [])
    ai_tr, ai_va = _patient_split(aisd_all, val_ratio, seed + 2) if aisd_all else ([], [])

    train_samples = ct_tr + bh_tr + ai_tr
    val_samples   = ct_va + bh_va + ai_va

    def _dist(ss):
        c = np.bincount([s[2] for s in ss], minlength=3)
        return f"bg={c[0]}, ischemic={c[1]}, hemorrhagic={c[2]}"
    print(f"  세그 학습: {len(train_samples)}개 ({_dist(train_samples)})")
    print(f"  세그 검증: {len(val_samples)}개 ({_dist(val_samples)})")

    train_ds = Seg3ClassDataset(train_samples, image_size, "train")
    val_ds   = Seg3ClassDataset(val_samples,   image_size, "val")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader
