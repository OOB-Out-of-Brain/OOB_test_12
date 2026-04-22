# Brain Stroke AI — CT 출혈 분석 파이프라인

뇌 CT 영상에서 **출혈 여부를 이진 분류(normal / hemorrhagic)** 하고, 출혈이 있으면 **병변 위치를 분할(segmentation)** 하는 end-to-end 파이프라인입니다.

- **Classifier**: EfficientNet-B2 (2-class)
- **Segmentor**: U-Net + ResNet34 encoder (binary mask)
- **Device**: Apple Silicon MPS / NVIDIA CUDA / CPU 자동 감지
- **라벨 전략**: 여러 다국적 데이터셋의 라벨을 `normal(0)` / `hemorrhagic(1)` 이진 체계로 통합

---

## Quick Start

```bash
# 1. 레포 클론
git clone https://github.com/OOB-Out-of-Brain/OOB_test_006.git
cd OOB_test_006

# 2. 가상환경 + 패키지 설치
python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. 데이터 자동 다운로드 (약 3.3GB, 30분~1시간)
python scripts/download_data.py

# 4. 학습 (분류기 + 분할기, 각각 따로 실행)
python training/train_classifier.py
python training/train_segmentor.py

# 5. 추론 (학습 끝난 후)
python demo.py --image path/to/ct.jpg
```

Python 3.10+, PyTorch 2.0+ 필요. **상세 가이드는 [HOWTRAIN.md](HOWTRAIN.md) 참조**.

---

## 사용 데이터셋 (4개)

| 데이터셋 | 출처 | 용량 | 역할 |
|---|---|---|---|
| **tekno21** | HuggingFace `BTX24/tekno21-brain-stroke-dataset-multi` | ~560MB | 분류 (normal/hemorrhagic) |
| **CT Hemorrhage** | PhysioNet `ct-ich v1.0.0` | ~1.2GB | 분류 + 분할 마스크 |
| **AISD (synthetic)** | 로컬 생성 | ~110MB | 분할 보조 |
| **BHSD** | HuggingFace `WuBiao/BHSD` | ~1.4GB | 분류 + 분할 (출혈 subtype 5종 → binary 통합) |

모든 데이터는 `python scripts/download_data.py` 한 번에 받아지고, BHSD의 3D NIfTI는 brain window(`center=40, width=80 HU`) 적용 후 2D 슬라이스로 자동 변환됩니다.

**라벨 매핑:**
- tekno21 `İnme Yok` → normal, `Kanama` → hemorrhagic, `iskemi` → 제외
- CT Hemorrhage `No_Hemorrhage=1` → normal, 나머지 → hemorrhagic
- BHSD 라벨 `1~5` (EDH/IPH/IVH/SAH/SDH) → 전부 hemorrhagic (subtype 정보는 버림)
- Normal 샘플은 BHSD에 없으므로 기존 데이터셋에서만 사용

---

## 주요 폴더

```
OOB_test_5/
├─ HOWTRAIN.md                 # 상세 학습/실행 가이드
├─ README.md                   # 이 문서
├─ config.yaml                 # 하이퍼파라미터 + 데이터 경로
├─ demo.py                     # 추론 데모
│
├─ data/
│  ├─ combined_dataset.py      # 분류기 통합 로더 (CT+tekno21+BHSD)
│  ├─ ct_hemorrhage_dataset.py # 분할기 통합 로더 (CT+BHSD 마스크)
│  └─ raw/, processed/         # 데이터 (gitignore, 스크립트로 다운로드)
│
├─ models/
│  ├─ classifier.py            # EfficientNet-B2 wrapper
│  └─ segmentor.py             # U-Net (smp)
│
├─ training/
│  ├─ train_classifier.py
│  ├─ train_segmentor.py
│  └─ metrics.py               # Dice, IoU, DiceBCE loss
│
├─ inference/
│  ├─ pipeline.py              # StrokePipeline (cls → seg → overlay)
│  └─ visualization.py
│
├─ scripts/
│  ├─ download_data.py         # 전체 데이터 통합 다운로드 (메인)
│  ├─ download_bhsd.py         # BHSD만 개별 다운로드
│  ├─ preprocess_bhsd.py       # NIfTI → 2D PNG 변환
│  ├─ generate_synthetic_aisd.py
│  └─ validate.py              # 학습 후 리포트
│
├─ checkpoints/                # 학습 결과 (gitignore)
├─ results/                    # 추론 결과
└─ archive/                    # 이전 실험 결과 보관
```

---

## 예상 소요 시간 (MacBook Apple Silicon 기준)

| 단계 | 시간 |
|---|---|
| 데이터 다운로드 + 전처리 | 30분~1시간 |
| 분류기 학습 (50 epoch) | ~1.5~2시간 |
| 분할기 학습 (early stopping) | ~20~40분 |
| 추론 1장 | 1~2초 |

NVIDIA GPU(CUDA)에서는 훨씬 빠릅니다.

---

## 테스트 / 추론 사용법

### 단일 이미지 추론

```bash
python demo.py --image path/to/ct.jpg
# 결과: results/{파일명}_result.png 에 저장됨

python demo.py --image input.jpg --output my_result.png
```

출력: `원본 | 결과 이미지(출혈 위치 오버레이) + 분류 확률 차트`

### 폴더 배치 테스트

폴더 안의 모든 이미지 자동 추론:

```bash
python scripts/run_batch_test.py --input-dir path/to/folder --output-dir results/
```

### 판독 로직 (pipeline)

1. 분류기가 normal/hemorrhagic 확률 출력
2. 분할기가 병변 마스크 생성
3. **최종 판독 규칙**:
   - 병변 비율 **≤ 1%** → `normal` (미세 오탐 무시)
   - 병변 비율 **> 1%** & 분류기 normal → `hemorrhagic` 으로 override (분류기 놓친 경우 세그가 보완)
   - 나머지 → 분류기 결과 유지

이 규칙은 [`inference/pipeline.py`](inference/pipeline.py) 에서 조정 가능.

---

## 검증 리포트

학습 완료 후 precision/recall/F1 + Dice/IoU:

```bash
python scripts/validate.py
```

---

## 트러블슈팅

- **PhysioNet zip 다운로드 실패** → 브라우저로 https://physionet.org/content/ct-ich/1.0.0/ 수동 다운로드 후 `data/raw/ct_hemorrhage/` 에 압축 해제
- **GPU OOM** → `config.yaml` 에서 `batch_size` 반으로 감소
- **MPS 디바이스 미지원** → 자동으로 CPU로 동작, 다만 느려짐
- 더 자세한 내용은 [HOWTRAIN.md](HOWTRAIN.md) 섹션 7 참조

---

## 라이선스 / 데이터 사용

각 데이터셋의 원래 라이선스를 따르세요:
- BHSD: **NC-ND (비상업적 사용만)**
- PhysioNet CT-ICH: 연구 목적 OK, 재배포 제한
- tekno21: TEKNOFEST 2021 공개 데이터
