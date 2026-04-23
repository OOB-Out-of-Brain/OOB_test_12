# Brain Stroke AI — CT 뇌졸중 분석 (2-class / 3-class 양방 지원)

뇌 CT 이미지를 받아 **정상 / 허혈성(ischemic) / 출혈성(hemorrhagic)** 판독과 병변 위치 시각화를 수행.
이 레포지토리는 두 가지 학습 경로를 모두 지원한다.

| 모드 | 분류 클래스 | 분할 클래스 | 체크포인트 폴더 | Val 기준 성능 |
|---|---|---|---|---|
| **2-class (A 규칙, 이전 버전)** | normal / hemorrhagic | 출혈 binary | `checkpoints/classifier/`, `checkpoints/segmentor/` | Val Acc 96.03%, Sens 93.32%, Spec 97.64% |
| **3-class (신규, tekno21 체계)** | normal / ischemic / hemorrhagic | bg / ischemic / hemorrhagic (softmax) | `checkpoints/classifier_3class/`, `checkpoints/segmentor_3class/` | 학습 중 |

- **Classifier**: EfficientNet-B2
- **Segmentor**: U-Net + ResNet34 encoder
- **Device**: Apple Silicon MPS / NVIDIA CUDA / CPU 자동 감지
- **Pipeline 규칙 (A 규칙)**: 분류기 softmax 결과 그대로 사용, 세그멘터 마스크는 시각화 전용

---

## 1. Quick Start

### 공통 준비 (clone → 학습 직전까지 한 번에)
```bash
git clone https://github.com/OOB-Out-of-Brain/OOB_test_10.git
cd OOB_test_10

python3 -m venv venv
source venv/bin/activate               # Windows: venv\Scripts\activate
pip install -r requirements.txt

python scripts/setup_all.py            # 학습용 데이터 4종 ~3.3GB 자동 다운로드
# CQ500 외부 테스트셋까지 같이 받으려면:
python scripts/setup_all.py --with-cq500   # +28GB, aria2c 필요
```
`setup_all.py` 내부에서 호출되는 스크립트:
- `scripts/download_data.py` — tekno21(HF) / CT Hemorrhage(PhysioNet) / AISD synthetic(로컬 생성) / BHSD(HF) 전처리까지 자동 처리
- `scripts/download_cq500.py` — (`--with-cq500` 지정 시) Academic Torrents 자동

### 2-class 학습 경로 (기존)
```bash
python training/train_classifier.py    # ~1.5~2시간
python training/train_segmentor.py     # ~30분
python demo.py --image path/to/ct.jpg
```

### 3-class 학습 경로 (신규)
```bash
python training/train_classifier_3class.py --epochs 100   # ≈ 3시간 20분 (MPS)
python training/train_segmentor_3class.py  --epochs 80    # ≈ 1시간
python scripts/run_batch_test_3class.py --input-dir /path/to/images --output-dir results/my_run
```

> 체크포인트(.pth)는 용량이 커서 레포에 포함되지 않는다. 위 학습 스크립트를
> 돌리면 각각 `checkpoints/classifier_3class/`, `checkpoints/segmentor_3class/`
> 경로에 자동 저장된다.

---

## 2. 데이터셋 가이드

### 2-1. 학습용 (필수) — `scripts/download_data.py` 한 번

| 데이터셋 | 출처 | 용량 | 2-class 역할 | 3-class 역할 |
|---|---|---|---|---|
| **tekno21** | HF `BTX24/tekno21-brain-stroke-dataset-multi` | ~560MB | 분류 (iskemi 제외) | 분류 (iskemi 포함) |
| **CT Hemorrhage** | PhysioNet `ct-ich v1.0.0` | ~1.2GB | 분류 + 분할 | 분류(normal/hem) + 분할(hem) |
| **AISD (synthetic)** | 로컬 생성 | ~110MB | 분할 보조 | 분할 **허혈** 소스 |
| **BHSD** | HF `WuBiao/BHSD` | ~1.4GB | 분류 + 분할 (subtype 5종 → binary) | 분류 + 분할 (→ hemorrhagic) |

**BHSD 처리**: 3D NIfTI → brain window (center 40, width 80 HU) → 2D PNG 슬라이스.
5개 subtype(EDH/IPH/IVH/SAH/SDH)은 모두 hemorrhagic으로 통합.

#### 3-class 라벨 매핑 요약
```
tekno21:    0 Kanama     → hemorrhagic (2)
            1 iskemi     → ischemic    (1)   ← 2-class에서는 제외됐지만 3-class는 포함
            2 İnme Yok   → normal      (0)
CT Hemorrhage: No_Hemorrhage=1 → normal (0),  =0 → hemorrhagic (2)
BHSD:       모든 슬라이스 → hemorrhagic (2)
AISD:       분할 마스크 → ischemic (1, 세그멘터만 사용)
```

### 2-2. 외부 테스트셋 (학습 금지, 평가 전용)

| 데이터셋 | 출처 | 용량 | 자동 | 사용처 |
|---|---|---|---|---|
| **CQ500** | qure.ai | ~28GB | ✅ `aria2c` 설치 후 | `evaluate_cq500.py` / `evaluate_cq500_3class.py` |

```bash
brew install aria2                        # macOS / Ubuntu는 sudo apt install aria2
python scripts/download_cq500.py          # Academic Torrents 자동 (28GB, reads.csv 포함)
```
aria2c가 불가능하면 Kaggle 대안 (40GB, `pip install kaggle` + `~/.kaggle/kaggle.json`):
```bash
python scripts/download_cq500.py --method kaggle
```
**라이선스**: CC BY-NC-SA 4.0 — 연구/평가 용도만, 학습 금지.
**GT 제약**: CQ500은 ICH(출혈) 라벨만 제공 → 3-class 평가 시에도 sens/spec은 hemorrhagic vs non-hemorrhagic 이진 기준.

---

## 3. 학습 실행

### 3-A. 2-class 학습 (기존)

```bash
python training/train_classifier.py                          # 기본 50 epoch
python training/train_classifier.py --epochs 30 --batch_size 8
python training/train_segmentor.py
```
체크포인트: `checkpoints/classifier/best_classifier.pth`, `checkpoints/segmentor/best_segmentor.pth`

### 3-B. 3-class 학습 (신규)

```bash
# 분류기: tekno21 + CT Hemorrhage + BHSD 전부 결합 (iskemi 제거 없음)
python training/train_classifier_3class.py --epochs 100

# tekno21만 쓰고 싶으면
python training/train_classifier_3class.py --epochs 100 --tekno21-only

# 세그멘터: CT Hemorrhage(출혈) + BHSD(출혈) + AISD(허혈) 결합, softmax 3-class
python training/train_segmentor_3class.py --epochs 80
```

체크포인트: `checkpoints/classifier_3class/best_classifier_3class.pth`,
`checkpoints/segmentor_3class/best_segmentor_3class.pth`

**참고 (MacBook M 시리즈 기준 측정값)**

| 구간 | 속도 |
|---|---|
| 3-class 분류기 학습 (batch=16, img=224, 9710장) | 1 epoch ≈ **2분** → 100 epoch ≈ **3시간 20분** |
| 3-class 분류기 검증 (2361장) | 1회 ≈ 8초 |
| 3-class 세그멘터 학습 (batch=8, img=256) | 1 epoch ≈ 1~1.5분 |

---

## 4. 테스트 실행 방법

### 4-A. 2-class 경로 (기존 체크포인트)

| 명령 | 용도 |
|---|---|
| `python demo.py --image path/to/ct.jpg` | 단일 이미지 |
| `python scripts/run_batch_test.py --input-dir ... --output-dir ...` | 폴더 배치 |
| `python scripts/evaluate_valset.py` | Val set 상세 (FP/FN 샘플) |
| `python scripts/evaluate_valset_compare.py` | A/B 규칙 비교 |
| `python scripts/save_all_valset_results.py` | 2089장 4폴더 분류 저장 |
| `python scripts/evaluate_cq500.py` | CQ500 외부 평가 |

### 4-B. 3-class 경로 (신규 체크포인트)

모두 `checkpoints/classifier_3class/best_classifier_3class.pth`, 선택적으로
`checkpoints/segmentor_3class/best_segmentor_3class.pth` 를 사용.
세그멘터 체크포인트가 없으면 분류만 수행하고 overlay는 생략된다.

#### 폴더 배치 추론 (brain_test 등)
```bash
python scripts/run_batch_test_3class.py \
    --input-dir /path/to/images \
    --output-dir results/my_3class_run/

# 체크포인트 위치를 바꾸고 싶으면
python scripts/run_batch_test_3class.py \
    --input-dir ... --output-dir ... \
    --cls-ckpt checkpoints/classifier_3class/best_classifier_3class.pth \
    --seg-ckpt checkpoints/segmentor_3class/best_segmentor_3class.pth

# 파일명에서 GT를 추측하지 않고 예측만 출력
python scripts/run_batch_test_3class.py --input-dir ... --output-dir ... --no-gt-from-name
```
파일명에 `nomal/normal`, `iskemi/ischem`, `EDH/ICH/SAH/SDH/hemorr` 등이 들어있으면 자동으로
GT를 추출해 3×3 confusion matrix와 정확도를 함께 출력한다.
출력: 각 이미지의 `{stem}_result.png` (원본 + 예측 오버레이) + 터미널 요약.

#### Val set 전체 3-class 상세 평가
```bash
python scripts/evaluate_valset_3class.py
# → results/valset_3class/
#     ├─ metrics.txt           (3×3 confusion matrix, per-class precision/recall)
#     ├─ summary.csv           (파일별 GT/예측/3개 확률/병변 면적)
#     └─ errors/gt_X_pred_Y/   (오분류 버킷별 최대 20장 시각화)
```

#### CQ500 외부 평가 (3-class)
```bash
python scripts/evaluate_cq500_3class.py
# → results/cq500_3class/
#     ├─ metrics.txt (3-class 예측 분포 + hemorrhagic vs non-hem 이진 sens/spec)
#     ├─ summary.csv
#     └─ false_positives/*.png
```
스캔 단위 판정 규칙:
- 슬라이스 중 하나라도 `hemorrhagic` → 스캔 = hemorrhagic
- 아니면 `ischemic` 슬라이스가 일정 수 이상이면 ischemic, 아니면 normal

---

## 5. Pipeline 판독 규칙 (A 규칙)

```
[이미지] → [분류기] → softmax 확률 → 최종 판독
       ↓ (선택, 시각화용)
       [세그멘터] → 3-class 마스크 → overlay 이미지
                  (ischemic=파란톤, hemorrhagic=빨간톤)
```
- 분류기 판단을 그대로 신뢰. 세그멘터는 "어디에 병변인가?" 위치만 시각화.
- 1% threshold 같은 후처리는 사용하지 않음 (OOB_test_6 참고 — FN 급증).

### 2-class 기준 성능 (val set 2089장)

| 지표 | A 규칙 (이 레포) | B 규칙 (OOB_test_6, 1% threshold) |
|---|---:|---:|
| Accuracy | **96.03%** | 72.00% |
| Sensitivity (출혈 탐지) | **93.32%** | 28.02% ⚠️ |
| Specificity (정상 식별) | 97.64% | **98.09%** |
| FP | 31 | **25** |
| FN (임상 위험) | **52** | 560 ⚠️ |

---

## 6. 폴더 구조

```
OOB_test_10/
├─ README.md                    # 이 문서
├─ HOWTRAIN.md                  # 학습 상세 가이드
├─ config.yaml                  # 하이퍼파라미터
├─ demo.py                      # 단일 이미지 추론 (2-class)
│
├─ data/
│  ├─ combined_dataset.py          # 2-class 분류 로더
│  ├─ combined_dataset_3class.py   # 3-class 분류 로더 (tekno21 + CT + BHSD)
│  ├─ seg_3class_dataset.py        # 3-class 분할 로더 (CT + BHSD + AISD)
│  ├─ ct_hemorrhage_dataset.py     # 2-class 분할 로더
│  ├─ classifier_dataset.py        # tekno21 단독 3-class 로더
│  └─ raw/, processed/             # .gitignore (download_data.py로 받음)
│
├─ models/
│  ├─ classifier.py                # EfficientNet-B2 (num_classes 동적)
│  ├─ segmentor.py                 # U-Net 1-채널 (2-class용)
│  └─ segmentor_3class.py          # U-Net 3-채널 softmax
│
├─ training/
│  ├─ train_classifier.py          # 2-class 분류 학습
│  ├─ train_classifier_3class.py   # 3-class 분류 학습 (신규)
│  ├─ train_segmentor.py           # 2-class 분할 학습
│  ├─ train_segmentor_3class.py    # 3-class 분할 학습 (Dice + CE)
│  └─ metrics.py
│
├─ inference/
│  ├─ pipeline.py                  # 2-class 추론 파이프라인
│  ├─ pipeline_3class.py           # 3-class 추론 파이프라인 (신규)
│  └─ visualization.py
│
├─ scripts/
│  ├─ setup_all.py                 # ⭐ 원샷 데이터 셋업 (신규)
│  ├─ download_data.py             # 학습용 4개 데이터셋
│  ├─ download_bhsd.py / preprocess_bhsd.py
│  ├─ generate_synthetic_aisd.py   # AISD 합성 데이터 (허혈 마스크)
│  ├─ download_cq500.py
│  ├─ run_batch_test.py            # 2-class 배치
│  ├─ run_batch_test_3class.py     # 3-class 배치 (신규)
│  ├─ evaluate_valset.py           # 2-class val 평가
│  ├─ evaluate_valset_3class.py    # 3-class val 평가 (신규)
│  ├─ evaluate_valset_compare.py
│  ├─ save_all_valset_results.py
│  ├─ evaluate_cq500.py            # 2-class CQ500 평가
│  └─ evaluate_cq500_3class.py     # 3-class CQ500 평가 (신규)
│
├─ checkpoints/                 # 학습 결과 (.gitignore)
│  ├─ classifier/                  # 2-class 분류기
│  ├─ segmentor/                   # 2-class 분할기
│  ├─ classifier_3class/           # 3-class 분류기
│  └─ segmentor_3class/            # 3-class 분할기
│
├─ results/                     # 추론 결과 (.gitignore)
└─ logs/                        # 학습 로그 (.gitignore)
```

---

## 7. 예상 소요 시간 (MacBook M 시리즈)

| 작업 | 시간 |
|---|---|
| `download_data.py` (3.3GB) | 30분 ~ 1시간 |
| 2-class 분류 학습 (50 epoch) | 1.5 ~ 2시간 |
| 3-class 분류 학습 (100 epoch) | ≈ 3시간 20분 |
| 2-class 분할 학습 (early stop) | 20 ~ 40분 |
| 3-class 분할 학습 (80 epoch) | 1 ~ 1.5시간 |
| 단일 이미지 추론 | 1 ~ 2초 |
| Val set 전체 평가 (≈2000장) | ≈ 10분 |
| CQ500 491 스캔 평가 | 30 ~ 60분 |

---

## 8. 트러블슈팅

- **PhysioNet zip 실패** → https://physionet.org/content/ct-ich/1.0.0/ 수동 다운로드 후 `data/raw/ct_hemorrhage/` 에 압축 해제
- **BHSD 다운로드 느림** → `huggingface-cli login`
- **CQ500 자동 실패** → 이메일 등록 방식이라 정상 (`--method kaggle` 또는 수동)
- **GPU OOM** → `config.yaml` 의 `batch_size` 절반
- **MPS 미지원** → 자동 CPU 전환 (느림)
- **3-class 체크포인트 없음 에러** → `training/train_classifier_3class.py` 먼저 실행
- **세그멘터 ckpt 없이 3-class 배치 돌림** → 분류만 수행, overlay 생략 (정상 동작)

---

## 9. 관련 레포

- **`OOB_test_6`**  : 1% threshold 규칙 버전 (스크리닝용, specificity ↑)
- **`OOB_test_7`**  : A 규칙 2-class 버전 (분류기 단독)
- **`OOB_test_10`** (이 레포) : 2-class + 3-class 양방 지원
