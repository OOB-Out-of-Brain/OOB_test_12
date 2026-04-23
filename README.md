# Brain Stroke AI — CT 뇌졸중 3-class 분석

뇌 CT 이미지를 받아 **정상 / 허혈성(ischemic) / 출혈성(hemorrhagic)** 판독과 병변 위치 시각화를 수행.

- **Classifier**: EfficientNet-B2 (3-class)
- **Segmentor**: U-Net + ResNet34 encoder (bg / ischemic / hemorrhagic, softmax 3채널)
- **Device**: Apple Silicon MPS / NVIDIA CUDA / CPU 자동 감지
- **Pipeline 규칙**: 분류기 softmax 결과 그대로 사용 (post-processing 없음), 세그멘터 마스크는 위치 시각화 전용

---

## 1. Quick Start

```bash
git clone https://github.com/OOB-Out-of-Brain/OOB_test_10.git
cd OOB_test_10

python3 -m venv venv
source venv/bin/activate                 # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 데이터 자동 다운로드 (tekno21 / CT Hemorrhage / AISD / BHSD, ~3.3GB)
python scripts/setup_all.py
# CQ500 외부 테스트셋까지 (+28GB, aria2c 필요):
# python scripts/setup_all.py --with-cq500

# 학습 (MPS 기준 50 epoch ≈ 1시간 40분)
python training/train_classifier.py --epochs 50
python training/train_segmentor.py  --epochs 80

# 추론
python demo.py --image path/to/ct.png
python scripts/run_batch_test.py --input-dir imgs/ --output-dir results/my_run
```

> 체크포인트(.pth)는 용량이 커서 레포에 포함되지 않는다. 학습 스크립트가 각각
> `checkpoints/classifier/` 와 `checkpoints/segmentor/` 에 자동 저장한다.

---

## 2. 데이터셋 가이드

### 2-1. 학습용 (필수) — `scripts/setup_all.py` 한 번

| 데이터셋 | 출처 | 용량 | 역할 | 3-class 매핑 |
|---|---|---|---|---|
| **tekno21** | HF `BTX24/tekno21-brain-stroke-dataset-multi` | ~560MB | 분류 | 0 Kanama→**hemorrhagic**, 1 iskemi→**ischemic**, 2 İnme Yok→**normal** |
| **CT Hemorrhage** | PhysioNet `ct-ich v1.0.0` | ~1.2GB | 분류 + 분할 | `No_Hemorrhage=1`→normal, `=0`→hemorrhagic |
| **AISD (synthetic)** | 로컬 생성 | ~110MB | 분할 (ischemic 마스크) | 분할 허혈 소스 |
| **BHSD** | HF `WuBiao/BHSD` | ~1.4GB | 분류 + 분할 (subtype 5종 통합) | 전부 → hemorrhagic |

**BHSD 처리**: 3D NIfTI → brain window (center 40, width 80 HU) → 2D PNG 슬라이스.
5개 subtype(EDH / IPH / IVH / SAH / SDH)은 모두 `hemorrhagic` 으로 통합.

### 2-1-a. PhysioNet 인증 필요 경고

PhysioNet은 2024년부터 anonymous zip 다운로드를 차단했다. 자동 다운로드 경로:

1. `PHYSIONET_USER` / `PHYSIONET_PASS` 환경변수가 있으면 Basic Auth로 시도
2. 없으면 Kaggle 미러(`cjinny/ct-ich-raw`, `pip install kaggle` + `~/.kaggle/kaggle.json`) 자동 fallback
3. 둘 다 실패 시 수동 경로 안내 (PhysioNet 로그인 후 zip 받아 `data/raw/ct_hemorrhage/ct_hemorrhage.zip` 로 복사)

```bash
# PhysioNet 인증 예시
export PHYSIONET_USER=<username>
export PHYSIONET_PASS=<password>
python scripts/setup_all.py
```

### 2-2. 외부 테스트셋 (학습 금지, 평가 전용)

| 데이터셋 | 출처 | 용량 | 용도 |
|---|---|---|---|
| **CQ500** | qure.ai / Academic Torrents | ~28GB | `scripts/evaluate_cq500.py` 외부 일반화 평가 |

```bash
brew install aria2                        # macOS (Ubuntu: sudo apt install aria2)
python scripts/download_cq500.py          # Academic Torrents 자동 (reads.csv 포함)
# Kaggle 대안 (~40GB): python scripts/download_cq500.py --method kaggle
```

**주의**: CQ500은 ICH(출혈) 라벨만 제공 → 3-class 모델 평가 시 hemorrhagic vs non-hemorrhagic 이진 지표로 환산.
라이선스: CC BY-NC-SA 4.0 (연구/평가 전용, 학습 금지).

---

## 3. 학습

### 3-1. 분류기 (3-class)

```bash
python training/train_classifier.py --epochs 50                  # 기본 3-class 결합 학습 (tekno21 + CT + BHSD)
python training/train_classifier.py --epochs 100 --batch_size 8  # 느린 머신용
python training/train_classifier.py --tekno21-only --epochs 50   # 터키 데이터만 사용
```
체크포인트: `checkpoints/classifier/best_classifier.pth`

### 3-2. 세그멘터 (3-class, bg/ischemic/hemorrhagic)

```bash
python training/train_segmentor.py --epochs 80
```
체크포인트: `checkpoints/segmentor/best_segmentor.pth`

### 3-3. 실시간 학습 모니터링 (새 터미널)

```bash
python scripts/watch_training.py            # logs/ 최신 .log 자동
python scripts/watch_training.py --no-bar   # 진행바 감추고 epoch 결과만
```

진행바 + Train/Val loss·acc + best 추적 + ETA 표시. `Ctrl+C`로 나가도 학습은 계속됨.

### 3-4. 예상 소요 시간 (MacBook M 시리즈, MPS)

| 작업 | 시간 |
|---|---|
| `setup_all.py` (3.3GB 데이터) | 30분 ~ 1시간 |
| 분류기 학습 (50 epoch, 9710장 × batch 16) | ≈ 1시간 40분 |
| 세그멘터 학습 (80 epoch) | 1 ~ 1.5시간 |
| 단일 이미지 추론 | 1 ~ 2초 |
| Val set 전체 평가 (≈2361장) | ≈ 10분 |
| CQ500 491 스캔 평가 | 30 ~ 60분 |

---

## 4. 추론 / 평가

### 4-1. 단일 이미지
```bash
python demo.py --image path/to/ct.png
# → results/{파일명}_result.png (원본 + 3-class 확률 바 + 병변 오버레이)
```

### 4-2. 폴더 배치 (brain_test 등)
```bash
python scripts/run_batch_test.py --input-dir /path/to/images --output-dir results/my_run

# 파일명에서 GT 추측 끄고 예측만
python scripts/run_batch_test.py --input-dir ... --output-dir ... --no-gt-from-name
```

파일명에 `nomal/normal`, `iskemi/ischem`, `EDH/ICH/SAH/SDH/hemorr` 등이 있으면 GT로 자동 인식해
3×3 confusion matrix와 정확도도 같이 출력.

### 4-3. Val set 전체 3-class 평가
```bash
python scripts/evaluate_valset.py
# → results/valset_3class/
#     ├─ metrics.txt      (3×3 confusion matrix + per-class precision/recall)
#     ├─ summary.csv      (샘플별 GT/예측/확률/병변 면적)
#     └─ errors/gt_X_pred_Y/  (오분류 버킷별 최대 20장 시각화)
```

### 4-4. CQ500 외부 평가
```bash
python scripts/evaluate_cq500.py
# → results/cq500_3class/
#     ├─ metrics.txt      (3-class 예측 분포 + hemorrhagic vs non-hem sens/spec)
#     ├─ summary.csv
#     └─ false_positives/
```

---

## 5. Pipeline 판독 규칙

```
[이미지] → [분류기] → softmax 확률 → 최종 판독 (argmax)
       ↓ (선택, 시각화용)
       [세그멘터] → 3-class 마스크 → overlay 이미지
                   (ischemic=파란톤, hemorrhagic=빨간톤)
```
- 분류기 판단을 그대로 신뢰, 1% threshold 같은 후처리 없음.
- 세그멘터는 "어디에 병변인가?" 위치 시각화 용도.

---

## 6. 폴더 구조

```
OOB_test_10/
├─ README.md                   # 이 문서
├─ HOWTRAIN.md                 # 학습 상세 가이드
├─ config.yaml                 # 하이퍼파라미터 (num_classes=3)
├─ demo.py                     # 단일 이미지 추론
│
├─ data/
│  ├─ combined_dataset.py      # 3-class 분류 로더 (tekno21 + CT + BHSD)
│  ├─ seg_dataset.py           # 3-class 분할 로더 (CT + BHSD + AISD)
│  └─ raw/, processed/         # .gitignore (setup_all.py 로 받음)
│
├─ models/
│  ├─ classifier.py            # EfficientNet-B2 (num_classes 파라미터)
│  └─ segmentor.py             # U-Net 3-채널 softmax
│
├─ training/
│  ├─ train_classifier.py      # 3-class 분류 학습
│  ├─ train_segmentor.py       # 3-class 분할 학습 (Dice + CE)
│  └─ metrics.py
│
├─ inference/
│  ├─ pipeline.py              # 3-class 추론 파이프라인
│  └─ visualization.py         # 3-class overlay 시각화
│
├─ scripts/
│  ├─ setup_all.py             # ⭐ 원샷 데이터 셋업
│  ├─ download_data.py         # 학습용 4개 데이터셋 (PhysioNet auth / Kaggle fallback)
│  ├─ download_bhsd.py / preprocess_bhsd.py
│  ├─ generate_synthetic_aisd.py
│  ├─ download_cq500.py        # CQ500 자동
│  ├─ run_batch_test.py        # 폴더 배치 추론
│  ├─ evaluate_valset.py       # Val set 상세 평가
│  ├─ evaluate_cq500.py        # CQ500 외부 평가
│  └─ watch_training.py        # 실시간 학습 모니터링
│
├─ checkpoints/                # 학습 결과 (.gitignore)
│  ├─ classifier/best_classifier.pth
│  └─ segmentor/best_segmentor.pth
│
├─ results/                    # 추론 결과 (.gitignore)
└─ logs/                       # 학습 로그 (.gitignore)
```

---

## 7. 트러블슈팅

- **PhysioNet 401/403** → 2-1-a 참조 (자격증명 설정 또는 Kaggle 사용)
- **BHSD 다운로드 느림** → `huggingface-cli login`
- **CQ500 토렌트 실패** → `--method kaggle` 로 대체
- **GPU OOM** → `config.yaml` 의 `batch_size` 절반
- **MPS 미지원** → 자동 CPU 전환 (느림)
- **체크포인트 없음 에러** → `training/train_classifier.py --epochs 50` 먼저 실행
- **세그멘터 ckpt 없이 배치 돌림** → 분류만 수행, overlay 생략 (정상 동작)
- **학습 중 파일 이름 바꾸면 DataLoader worker가 스크립트 경로를 못 찾아 터짐** → 학습 돌리는 동안 소스 rename 금지
