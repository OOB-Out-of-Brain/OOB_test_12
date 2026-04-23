"""모든 데이터셋 다운로드/준비 통합 스크립트.

준비하는 데이터셋:
  1. tekno21         : HuggingFace 자동 다운로드
  2. CT Hemorrhage   : PhysioNet에서 zip 직접 다운로드
  3. BHSD            : HuggingFace (별도 스크립트 호출)
  4. AISD (synthetic): 로컬 생성 (별도 스크립트)

실행:
    python scripts/download_data.py
"""

import sys, os, zipfile, subprocess, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


TEKNO21_CACHE   = Path("./data/raw/tekno21")
CT_HEM_DIR      = Path("./data/raw/ct_hemorrhage")
CT_HEM_UNPACKED = CT_HEM_DIR / "computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0"
# PhysioNet은 2024년 이후 anonymous 다운로드가 차단됨 → Basic Auth 필요.
# CT_HEM_ZIP_URL_ANON (legacy, 이제 redirect 후 401) 은 fallback 용으로만 유지.
CT_HEM_ZIP_URL_AUTH = "https://physionet.org/content/ct-ich/get-zip/1.0.0/"
CT_HEM_ZIP_URL_ANON = "https://physionet.org/static/published-projects/ct-ich/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0.zip"
CT_HEM_KAGGLE_REF   = "cjinny/ct-ich-raw"  # Kaggle mirror (~630MB zip)

AISD_DIR        = Path("./data/raw/aisd")

BHSD_DIR        = Path("./data/raw/bhsd/label_192")
BHSD_PROCESSED  = Path("./data/processed/bhsd")


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r    {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB")
        sys.stdout.flush()


# ── 1. tekno21 ───────────────────────────────────────────────────────────────
def check_tekno21() -> bool:
    print("\n[1] tekno21 (HuggingFace)")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "BTX24/tekno21-brain-stroke-dataset-multi",
            split="train",
            cache_dir=str(TEKNO21_CACHE),
        )
        print(f"  ✅ tekno21: {len(ds)}개 슬라이스")
        return True
    except Exception as e:
        print(f"  ❌ 다운로드 실패: {e}")
        return False


# ── 2. CT Hemorrhage (PhysioNet → Kaggle fallback) ───────────────────────────
def _download_with_auth(url: str, dest: Path, user: str, pwd: str) -> bool:
    """HTTP Basic Auth로 다운로드 (PhysioNet 방식)."""
    try:
        pm = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        pm.add_password(None, url, user, pwd)
        auth = urllib.request.HTTPBasicAuthHandler(pm)
        opener = urllib.request.build_opener(auth)
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()
        return True
    except Exception as e:
        print(f"\n    인증 다운로드 실패: {e}")
        return False


def _download_kaggle_ct_ich(dest_dir: Path) -> bool:
    """Kaggle 미러에서 다운로드. `~/.kaggle/kaggle.json` 필요."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("    kaggle 패키지 없음 → `pip install kaggle`")
        return False
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"    Kaggle 인증 실패 ({e}) — `~/.kaggle/kaggle.json` 확인")
        return False
    try:
        api.dataset_download_files(CT_HEM_KAGGLE_REF,
                                    path=str(dest_dir), unzip=True, quiet=False)
        return True
    except Exception as e:
        print(f"    Kaggle 다운로드 실패: {e}")
        return False


def check_ct_hemorrhage() -> bool:
    print("\n[2] CT Hemorrhage (PhysioNet)")
    csv_path = CT_HEM_UNPACKED / "hemorrhage_diagnosis.csv"
    if csv_path.exists():
        print(f"  ✅ 이미 있음: {CT_HEM_UNPACKED}")
        return True

    zip_path = CT_HEM_DIR / "ct_hemorrhage.zip"
    CT_HEM_DIR.mkdir(parents=True, exist_ok=True)

    # 경로 1: 환경변수로 PhysioNet 자격증명이 있으면 먼저 시도
    user = os.environ.get("PHYSIONET_USER")
    pwd = os.environ.get("PHYSIONET_PASS")
    if not zip_path.exists() and user and pwd:
        print(f"  [1/3] PhysioNet 인증 다운로드 시도 ($PHYSIONET_USER={user})")
        print(f"    URL: {CT_HEM_ZIP_URL_AUTH}")
        if _download_with_auth(CT_HEM_ZIP_URL_AUTH, zip_path, user, pwd):
            print(f"  ✅ 인증 다운로드 성공")

    # 경로 2: 익명 (과거엔 됐으나 2024~ 차단됨)
    if not zip_path.exists():
        print(f"  [2/3] 익명 다운로드 시도 (대부분 실패 예상)")
        print(f"    URL: {CT_HEM_ZIP_URL_ANON}")
        try:
            urllib.request.urlretrieve(CT_HEM_ZIP_URL_ANON, zip_path, reporthook=_progress)
            print()
        except Exception as e:
            print(f"\n    (예상대로) 익명 차단: {e}")
            if zip_path.exists() and zip_path.stat().st_size < 1024:
                zip_path.unlink()  # HTML 에러 페이지 삭제

    # 경로 3: Kaggle 미러 fallback
    if not zip_path.exists():
        print(f"  [3/3] Kaggle 미러로 전환 ({CT_HEM_KAGGLE_REF})")
        if _download_kaggle_ct_ich(CT_HEM_DIR):
            # Kaggle 압축은 이미 풀려있고 구조가 다를 수 있음 → 구조 확인
            if CT_HEM_UNPACKED.exists() and (CT_HEM_UNPACKED / "hemorrhage_diagnosis.csv").exists():
                print(f"  ✅ Kaggle 미러 완료: {CT_HEM_UNPACKED}")
                return True
            # Kaggle ref가 다른 구조로 압축 해제될 경우 hemorrhage_diagnosis.csv 위치 파악
            for csv_cand in CT_HEM_DIR.rglob("hemorrhage_diagnosis.csv"):
                root = csv_cand.parent
                if root != CT_HEM_UNPACKED:
                    print(f"    Kaggle 구조 조정: {root} → {CT_HEM_UNPACKED}")
                    CT_HEM_UNPACKED.mkdir(parents=True, exist_ok=True)
                    for item in root.iterdir():
                        target = CT_HEM_UNPACKED / item.name
                        if not target.exists():
                            item.rename(target)
                print(f"  ✅ Kaggle 미러 완료: {CT_HEM_UNPACKED}")
                return True

    if not zip_path.exists():
        print(f"""
  ❌ 자동 다운로드 전부 실패. 다음 중 하나 선택:
     (a) PhysioNet 계정 있으면 자격증명 설정 후 재시도
         export PHYSIONET_USER=<username>
         export PHYSIONET_PASS=<password>
         python scripts/download_data.py
     (b) Kaggle 설정 후 재시도
         pip install kaggle
         # ~/.kaggle/kaggle.json 배치 (Kaggle → Account → Create API Token)
         python scripts/download_data.py
     (c) 수동 다운로드
         https://physionet.org/content/ct-ich/1.0.0/ 로그인 후 zip 받아서
         → {zip_path} 로 복사하고 재실행""")
        return False

    print("  압축 해제 중...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(CT_HEM_DIR)
    print(f"  ✅ 완료: {CT_HEM_UNPACKED}")
    return True


# ── 3. AISD (synthetic) ──────────────────────────────────────────────────────
def check_aisd() -> bool:
    print("\n[3] AISD (synthetic)")
    images_dir = AISD_DIR / "images"
    masks_dir = AISD_DIR / "masks"
    if images_dir.exists() and len(list(images_dir.glob("*.png"))) > 0:
        print(f"  ✅ 이미 있음 ({len(list(images_dir.glob('*.png')))}개 이미지)")
        return True

    gen_script = Path(__file__).parent / "generate_synthetic_aisd.py"
    if gen_script.exists():
        print(f"  synthetic AISD 생성 중 (generate_synthetic_aisd.py)...")
        try:
            subprocess.run([sys.executable, str(gen_script)], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 생성 실패: {e}")
    print(f"""
  ⚠️ 합성 AISD 생성 스크립트 실패 또는 없음.
  실제 AISD는 https://github.com/GriffinLiang/AISD 에서 수동 요청.
""")
    return False


# ── 4. BHSD ──────────────────────────────────────────────────────────────────
def check_bhsd() -> bool:
    print("\n[4] BHSD (HuggingFace)")
    if BHSD_DIR.exists() and (BHSD_DIR / "images").exists():
        n = len(list((BHSD_DIR / "images").glob("*.nii.gz")))
        print(f"  ✅ 원본 이미 있음 ({n}개 볼륨)")
    else:
        print("  원본 다운로드 → scripts/download_bhsd.py 실행...")
        script = Path(__file__).parent / "download_bhsd.py"
        try:
            subprocess.run([sys.executable, str(script)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 실패: {e}")
            return False

    # 전처리된 PNG 슬라이스 확인
    if BHSD_PROCESSED.exists() and (BHSD_PROCESSED / "index.csv").exists():
        n_img = len(list((BHSD_PROCESSED / "images").glob("*.png")))
        print(f"  ✅ 전처리 완료 ({n_img}개 슬라이스)")
    else:
        print("  2D 슬라이스 전처리 → scripts/preprocess_bhsd.py 실행...")
        script = Path(__file__).parent / "preprocess_bhsd.py"
        try:
            subprocess.run([sys.executable, str(script)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 전처리 실패: {e}")
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  뇌졸중 AI 프로젝트 — 전체 데이터셋 준비")
    print("=" * 60)

    results = {
        "tekno21":       check_tekno21(),
        "CT_hemorrhage": check_ct_hemorrhage(),
        "AISD":          check_aisd(),
        "BHSD":          check_bhsd(),
    }

    print("\n" + "=" * 60)
    print("  요약")
    print("=" * 60)
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")

    if all(results.values()):
        print("\n모든 데이터 준비 완료. 학습 시작:")
        print("  python training/train_classifier.py")
        print("  python training/train_segmentor.py")
    else:
        print("\n일부 데이터셋 실패. 위 안내 확인 후 재실행하세요.")


if __name__ == "__main__":
    main()
