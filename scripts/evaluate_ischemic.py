"""Ischemic 전용 평가 (tekno21 iskemi 분류기-held-out 서브셋).

허혈 라벨(GT=ischemic)만 가진 테스트셋이 공개로 구하기 어려워서
tekno21 의 classifier train/val split (seed=42, val_ratio=0.2, stratified) 중
val 쪽의 iskemi 샘플만 뽑아서 평가한다. 이 샘플들은 분류기 학습에 쓰이지 않음.

(세그멘터는 동일 이미지에 Grad-CAM pseudo-mask 로 학습했지만,
 pseudo-mask 는 실제 GT 라벨이 아닌 CAM 히트맵이라 품질 평가용 프락시로 사용 가능.
 Top-level 진단은 분류기가 담당하므로 held-out 평가로서 유효함.)

출력 (results/ischemic_test/):
  - summary.csv           (idx, pred, confidence, isch_pct_brain, hem_pct_brain)
  - metrics.txt           (recall / 혼동행렬 / 평균 isch%)
  - correct/ischemic/                         (허혈 맞춘 것)
  - wrong/ischemic_to_normal/                 (허혈을 정상으로 놓친 것)
  - wrong/ischemic_to_hemorrhagic/            (허혈을 출혈로 혼동한 것)
"""

import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from inference.pipeline import StrokePipeline
from scripts._eval_common import classify_bucket, save_3panel, ensure_bucket_dirs


TEKNO21_CACHE = "./data/raw/tekno21"
OUT_DIR = Path("./results/ischemic_test")
ISKEMI_LABEL_ORIG = 1     # tekno21 원 라벨 (0=Kanama 1=iskemi 2=Inme Yok)
ISCHEMIC_REMAPPED = 1     # 파이프라인 3-class 라벨 (0=normal 1=ischemic 2=hemorrhagic)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 허혈 GT 만 존재 → correct/ischemic + wrong/ischemic_to_* 버킷만 의미 있음
    ensure_bucket_dirs(OUT_DIR, include_ischemic_correct=True)

    print("tekno21 로드 (캐시 경로:", TEKNO21_CACHE, ")")
    ds = load_dataset(
        "BTX24/tekno21-brain-stroke-dataset-multi",
        split="train",
        cache_dir=TEKNO21_CACHE,
    )

    # 분류기와 동일한 split 재현 (combined_dataset.py 의 로직 그대로)
    remap = {0: 2, 1: 1, 2: 0}
    tk_all = []
    for i in range(len(ds)):
        orig = int(ds[i]["label"])
        if orig not in remap:
            continue
        tk_all.append((i, remap[orig]))

    labels = [s[1] for s in tk_all]
    idx_list = list(range(len(tk_all)))
    _, val_i = train_test_split(
        idx_list, test_size=0.2, stratify=labels, random_state=42
    )
    # 이 중 ischemic 만
    iskemi_val = [tk_all[i][0] for i in val_i if tk_all[i][1] == ISCHEMIC_REMAPPED]
    print(f"  tekno21 전체: {len(tk_all)} | 분류기 val split: {len(val_i)} | "
          f"iskemi val 전용: {len(iskemi_val)}")

    pipe = StrokePipeline(
        classifier_ckpt="./checkpoints/classifier/best_classifier.pth",
        segmentor_ckpt="./checkpoints/segmentor/best_segmentor.pth"
            if Path("./checkpoints/segmentor/best_segmentor.pth").exists()
            else None,
    )

    class_names = ["normal", "ischemic", "hemorrhagic"]
    pred_counts = {c: 0 for c in class_names}
    isch_pct_list = []
    summary_rows = []

    for n, hf_idx in enumerate(iskemi_val):
        item = ds[hf_idx]
        im = item["image"]
        if not isinstance(im, Image.Image):
            im = Image.fromarray(im)
        img = np.array(im.convert("RGB"))

        r = pipe.run(img)
        pred = r.class_name
        pred_counts[pred] += 1
        if pred == "ischemic":
            isch_pct_list.append(r.ischemic_area_pct)

        summary_rows.append([
            hf_idx, pred,
            f"{r.confidence:.3f}",
            f"{r.class_probs.get('normal', 0):.3f}",
            f"{r.class_probs.get('ischemic', 0):.3f}",
            f"{r.class_probs.get('hemorrhagic', 0):.3f}",
            f"{r.ischemic_area_pct:.2f}",
            f"{r.hemorrhagic_area_pct:.2f}",
            r.brain_area_px,
        ])

        # GT 는 항상 ischemic
        bucket = classify_bucket("ischemic", pred, has_ischemic_gt=True)
        save_3panel(
            img, r,
            OUT_DIR / bucket / f"tk{hf_idx:05d}.png",
            "ischemic", dpi=100,
        )

        if (n + 1) % 50 == 0:
            acc_so_far = pred_counts["ischemic"] / (n + 1)
            print(f"  {n+1}/{len(iskemi_val)}  ischemic_recall={acc_so_far:.4f}")

    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tekno21_idx", "pred", "confidence",
                    "p_normal", "p_ischemic", "p_hemorrhagic",
                    "ischemic_pct_of_brain", "hemorrhagic_pct_of_brain", "brain_px"])
        w.writerows(summary_rows)

    total = len(iskemi_val)
    recall = pred_counts["ischemic"] / max(total, 1)
    mean_isch_pct = (sum(isch_pct_list) / len(isch_pct_list)) if isch_pct_list else 0.0

    report = f"""\
Ischemic 전용 평가 (tekno21 iskemi val 서브셋, 분류기-held-out)
=============================================
총 {total} 샘플 (GT=ischemic)

예측 분포:
  normal       : {pred_counts['normal']:>5}  ← 놓침 (ischemic_to_normal)
  ischemic     : {pred_counts['ischemic']:>5}  ← 정답 (correct/ischemic)
  hemorrhagic  : {pred_counts['hemorrhagic']:>5}  ← 병변 혼동 (ischemic_to_hemorrhagic)

Ischemic recall : {recall:.4f}
예측=ischemic 한 샘플의 평균 isch 면적 : {mean_isch_pct:.2f}% of brain

결과 버킷:
  {OUT_DIR}/correct/ischemic/                  (맞춘 샘플)
  {OUT_DIR}/wrong/ischemic_to_normal/          (놓친 샘플)
  {OUT_DIR}/wrong/ischemic_to_hemorrhagic/     (출혈로 혼동한 샘플)
"""
    print("\n" + report)
    (OUT_DIR / "metrics.txt").write_text(report)
    print(f"저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
