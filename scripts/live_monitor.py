"""
실시간 학습 모니터링
실행: python scripts/live_monitor.py
"""
import re
import time
from pathlib import Path
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.console import Group
from rich import box

SEG_LOG   = "/private/tmp/claude-501/-Users-bari-brain-stroke-ai/6b56275a-b1ae-4ede-8e23-7a21d5428975/tasks/buqzn31j6.output"
SEG_TOTAL = 80


def parse_seg(log_path):
    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s+\|\s+Train loss=([\d.]+)\s+dice=([\d.]+)"
        r"\s+\|\s+Val loss=([\d.]+)\s+dice=([\d.]+)\s+iou=([\d.]+)"
    )
    rows = []
    try:
        text = Path(log_path).read_text(errors="ignore")
        for m in pattern.finditer(text):
            rows.append({
                "epoch": int(m.group(1)),
                "tl": float(m.group(2)), "td": float(m.group(3)),
                "vl": float(m.group(4)), "vd": float(m.group(5)), "vi": float(m.group(6)),
            })
    except FileNotFoundError:
        pass
    return rows


def render(rows):
    current   = rows[-1]["epoch"] if rows else 0
    best_dice = max((r["vd"] for r in rows), default=0.0)
    best_ep   = next((r["epoch"] for r in rows if r["vd"] == best_dice), 0)
    pct       = current / SEG_TOTAL * 100
    bar_len   = 40
    filled    = int(bar_len * pct / 100)

    # ── 진행률 바 ──
    prog = Text()
    prog.append("\n  Epoch  ", style="bold")
    prog.append(f"{current:3d} / {SEG_TOTAL}", style="cyan bold")
    prog.append("   [", style="dim")
    prog.append("█" * filled, style="green bold")
    prog.append("░" * (bar_len - filled), style="dim")
    prog.append("]", style="dim")
    prog.append(f"  {pct:.0f}%\n", style="bold white")
    prog.append(f"  Best Val Dice: ", style="")
    prog.append(f"{best_dice:.4f}", style="bold green")
    prog.append(f"  (epoch {best_ep})\n", style="dim")

    # ── 에폭 테이블 ──
    table = Table(
        box=box.SIMPLE_HEAVY,
        border_style="blue",
        header_style="bold cyan",
        show_edge=True,
    )
    table.add_column("Epoch",      justify="right",  style="cyan",  width=7)
    table.add_column("Train Loss", justify="right",  width=11)
    table.add_column("Train Dice", justify="right",  width=11)
    table.add_column("Val Loss",   justify="right",  width=10)
    table.add_column("Val Dice",   justify="right",  width=10)
    table.add_column("Val IoU",    justify="right",  width=9)
    table.add_column(" ",          justify="center", width=3)

    for r in rows[-20:]:
        is_best = r["epoch"] == best_ep
        d_color = "green" if r["vd"] >= 0.5 else "yellow" if r["vd"] >= 0.3 else "red"
        table.add_row(
            f"[bold cyan]{r['epoch']}[/bold cyan]" if is_best else str(r["epoch"]),
            f"{r['tl']:.4f}",
            f"{r['td']:.4f}",
            f"{r['vl']:.4f}",
            f"[{d_color} bold]{r['vd']:.4f}[/{d_color} bold]",
            f"{r['vi']:.4f}",
            "[green]★[/green]" if is_best else "",
        )

    return Group(prog, table)


if __name__ == "__main__":
    print("  Segmentation 실시간 모니터링  (Ctrl+C 종료)\n")
    with Live(render([]), refresh_per_second=1) as live:
        while True:
            live.update(render(parse_seg(SEG_LOG)))
            time.sleep(3)
