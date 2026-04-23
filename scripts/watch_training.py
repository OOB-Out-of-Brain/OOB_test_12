"""학습 로그 실시간 모니터링 — tqdm 진행바 + epoch 요약 + ETA.

사용법:
    python scripts/watch_training.py                              # logs/ 에서 가장 최근 .log 자동
    python scripts/watch_training.py logs/train_3class_ep50.log   # 특정 파일

내부 구현: `tail -f`를 subprocess로 실행 + 출력을 \\r 기준으로 잘라 읽는다.
이 방식은 tqdm \\r 세그먼트를 즉시 잡을 수 있어, Python 수동 tail 보다 반응이 빠르다.

옵션:
    --no-bar     진행바 숨기고 epoch 완료 결과만 표시
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


CSI = "\033["
CLEAR = f"{CSI}2K\r"
GREEN = f"{CSI}32m"
CYAN = f"{CSI}36m"
YELLOW = f"{CSI}33m"
RED = f"{CSI}31m"
DIM = f"{CSI}2m"
BOLD = f"{CSI}1m"
RESET = f"{CSI}0m"

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s*\|\s*Train loss=([\d.]+)\s+acc=([\d.]+)"
    r"\s*\|\s*Val loss=([\d.]+)\s+acc=([\d.]+)"
)
TQDM_RE = re.compile(
    r"(train:|eval :)\s+(\d+)%\|[^|]*\|\s+(\d+)/(\d+).*?\[([^\]]+)\]"
)


def human(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def pick_latest_log() -> Path | None:
    logs = Path("logs")
    if not logs.exists():
        return None
    cands = sorted(logs.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def handle(seg: str, state: dict, show_bar: bool):
    s = seg.rstrip("\r\n").strip()
    if not s:
        return

    m_ep = EPOCH_RE.search(s)
    if m_ep:
        ep = int(m_ep.group(1)); tot = int(m_ep.group(2))
        tl = float(m_ep.group(3)); ta = float(m_ep.group(4))
        vl = float(m_ep.group(5)); va = float(m_ep.group(6))
        now = time.time()
        elapsed = now - state["start"]
        state["completed"].append((ep, tot, va))

        best_ep, best_va = max(state["completed"], key=lambda c: c[2])[::2]
        per_ep = elapsed / len(state["completed"])
        eta = per_ep * max(0, tot - ep)
        color = GREEN if va >= best_va else CYAN

        sys.stdout.write(CLEAR)
        print(f"{color}Ep {ep:>3}/{tot}{RESET}  "
              f"train loss={tl:.4f} acc={ta:.4f}  |  "
              f"{BOLD}val acc={va:.4f}{RESET} (loss={vl:.4f})  "
              f"best={best_va:.4f}(ep{best_ep})  "
              f"elapsed={human(elapsed)}  ETA={human(eta)}")
        sys.stdout.flush()
        state["last"] = ""
        return

    if "best val acc" in s or "best Dice" in s or "best lesion" in s:
        sys.stdout.write(CLEAR + f"  {YELLOW}↳ {s}{RESET}\n"); sys.stdout.flush()
        state["last"] = ""; return

    if "Early stopping" in s or "학습 완료" in s:
        sys.stdout.write(CLEAR + f"  {GREEN}{s}{RESET}\n"); sys.stdout.flush()
        state["last"] = ""; return

    if "Traceback" in s or re.match(r"^\w*Error", s):
        sys.stdout.write(CLEAR + f"  {RED}{s}{RESET}\n"); sys.stdout.flush()
        state["last"] = ""; return

    if not show_bar:
        return

    m = TQDM_RE.search(s)
    if not m:
        return
    phase = m.group(1).rstrip(":").strip()
    pct = int(m.group(2))
    cur = int(m.group(3)); tot = int(m.group(4))
    meta = m.group(5).strip()
    # 현재 진행 중 epoch = 완료된 수 + 1
    ep_n = state["completed"][-1][0] + 1 if state["completed"] else 1
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    msg = (f"{DIM}ep{ep_n:>2}{RESET} {phase:<5} "
           f"[{bar}] {pct:>3}%  {cur}/{tot}  {meta}")
    if msg != state["last"]:
        sys.stdout.write(CLEAR + msg)
        sys.stdout.flush()
        state["last"] = msg


def seed_state_from_log(log_path: Path, state: dict):
    """이미 진행된 epoch 라인을 로그에서 읽어 state["completed"]에 반영.
    세션 중간에 watcher를 붙여도 올바른 ep 번호/ETA 표시되게 함."""
    try:
        with open(log_path, "rb") as f:
            data = f.read()
    except Exception:
        return
    text = data.decode("utf-8", errors="ignore")
    # \r 도 개행으로 취급
    for line in re.split(r"[\r\n]+", text):
        m = EPOCH_RE.search(line)
        if m:
            ep = int(m.group(1)); tot = int(m.group(2))
            va = float(m.group(6))
            state["completed"].append((ep, tot, va))


def run(log_path: Path, show_bar: bool):
    print(f"{BOLD}watching{RESET}  {log_path}")
    print(f"{BOLD}start   {RESET}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 78)

    state = {"start": time.time(), "completed": [], "last": ""}
    seed_state_from_log(log_path, state)
    if state["completed"]:
        last_ep, last_tot, _ = state["completed"][-1]
        best_ep, best_va = max(state["completed"], key=lambda c: c[2])[::2]
        print(f"  {DIM}기존 완료 {len(state['completed'])} epoch — "
              f"현재 ep{last_ep + 1}/{last_tot} 진행 중, "
              f"best val acc={best_va:.4f}(ep{best_ep}){RESET}")

    # `tail -f -n 0`로 마지막 지점부터 새 바이트만 받음.
    # stdout을 binary로 받아 \r 세그먼트 단위로 쪼갬 (tqdm은 \r 로만 구분됨).
    proc = subprocess.Popen(
        ["tail", "-f", "-n", "0", str(log_path)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    assert proc.stdout is not None

    # os.read로 non-blocking 스러운 바이트 단위 읽기 + \r/\n 기준 세그먼트 추출.
    fd = proc.stdout.fileno()
    buf = b""
    try:
        while True:
            try:
                chunk = os.read(fd, 4096)
            except InterruptedError:
                continue
            if not chunk:
                time.sleep(0.05)
                continue
            buf += chunk
            # \r 또는 \n 기준으로 자르되, 마지막 미완성 세그먼트는 다음 iteration 에 합침.
            # 단, 마지막 \r 뒤의 내용이 비어 있으면(=즉시 커밋되는 상태) 바로 처리.
            segments = re.split(rb"[\r\n]", buf)
            buf = segments[-1]
            # 만약 chunk가 \r 로 끝났으면 segments[-1] = b'' — 앞 세그먼트를 완성 처리.
            for seg in segments[:-1]:
                handle(seg.decode("utf-8", errors="ignore"), state, show_bar)
            # 현재 미완성 세그먼트라도 tqdm 바 형태면 즉시 반영 (지연 최소화)
            if buf and show_bar:
                s = buf.decode("utf-8", errors="ignore")
                if ("train:" in s or "eval :" in s) and "]" in s:
                    handle(s, state, show_bar)
    except KeyboardInterrupt:
        sys.stdout.write(CLEAR)
        print(f"\n  {YELLOW}중단. 학습은 백그라운드에서 계속 진행 중.{RESET}")
    finally:
        try:
            proc.send_signal(signal.SIGTERM)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", help="로그 파일 경로 (미지정 시 logs/ 최신)")
    parser.add_argument("--no-bar", action="store_true", help="진행바 감추고 epoch 만")
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else pick_latest_log()
    if not log_path or not log_path.exists():
        sys.exit("  logs/ 에 .log 없음 — 학습 시작 후 다시 실행")
    run(log_path, show_bar=not args.no_bar)


if __name__ == "__main__":
    main()
