"""3-class 추론 결과 시각화.

PipelineResult (inference.pipeline) 객체를 받아 3-panel figure 생성:
  1. 원본 CT
  2. 3-class 확률 막대
  3. 병변 오버레이 (ischemic=파란톤, hemorrhagic=빨간톤)
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.family"] = "DejaVu Sans"


CLASS_TITLE_COLORS = {
    "normal":      (100, 220, 100),
    "ischemic":    ( 60, 120, 255),
    "hemorrhagic": (255,  80,  80),
}

ISCHEMIC_RGB    = (60, 120, 255)
HEMORRHAGIC_RGB = (255,  80,  80)

_BAR_PALETTE = {
    "normal":      "#4caf50",
    "ischemic":    "#2196f3",
    "hemorrhagic": "#f44336",
}


def visualize_result(orig_np: np.ndarray, result, alpha: float = 0.45) -> np.ndarray:
    fig = _build_figure(orig_np, result, alpha)
    arr = _fig_to_numpy(fig)
    plt.close(fig)
    return arr


def save_visualization(orig_np: np.ndarray, result, output_path: str,
                       alpha: float = 0.45, dpi: int = 150):
    fig = _build_figure(orig_np, result, alpha)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _build_figure(orig_np: np.ndarray, result, alpha: float):
    # 3-panel 고정 (원본 · 그래프 · overlay).
    # 중앙 그래프 패널은 두 단계: (위) 분류기 3-class 확률, (아래) 뇌 영역 내 구성 비율.
    has_lesion = (result.ischemic_area_px > 0) or (result.hemorrhagic_area_px > 0)

    fig = plt.figure(figsize=(18, 5.8), facecolor="black")
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.02, right=0.98, top=0.88, bottom=0.08, wspace=0.12, hspace=0.55,
    )

    ax_orig    = fig.add_subplot(gs[:, 0])
    ax_probs   = fig.add_subplot(gs[0, 1])
    ax_area    = fig.add_subplot(gs[1, 1])
    ax_overlay = fig.add_subplot(gs[:, 2])

    title_rgb = CLASS_TITLE_COLORS.get(result.class_name, (200, 200, 200))
    title_norm = [c / 255 for c in title_rgb]

    # 1. Original
    ax_orig.imshow(orig_np)
    ax_orig.set_title("Original CT", color="white", fontsize=13, pad=8)
    ax_orig.axis("off")

    # 2-a. Classifier probabilities (위)
    _style_bar_axes(ax_probs)
    keys = ["normal", "ischemic", "hemorrhagic"]
    probs = [result.class_probs.get(k, 0.0) for k in keys]
    colors = [_BAR_PALETTE[k] for k in keys]
    bars = ax_probs.barh([k.capitalize() for k in keys], probs, color=colors, height=0.55)
    for bar, p in zip(bars, probs):
        ax_probs.text(min(p + 0.01, 0.95), bar.get_y() + bar.get_height() / 2,
                      f"{p:.1%}", va="center", color="white",
                      fontsize=10, fontweight="bold")
    ax_probs.set_xlim(0, 1.0)
    ax_probs.set_xlabel("Classifier probability", color="white", fontsize=10)
    ax_probs.set_title(
        f"Diagnosis: {result.class_name.upper()} ({result.confidence:.1%})",
        color=title_norm, fontsize=12, fontweight="bold", pad=6,
    )

    # 2-b. Brain area composition (아래) — 뇌 영역 대비 병변 %
    _style_bar_axes(ax_area)
    normal_pct = result.normal_brain_pct
    isch_pct = result.ischemic_area_pct
    hem_pct = result.hemorrhagic_area_pct
    area_vals = [normal_pct, isch_pct, hem_pct]
    area_bars = ax_area.barh(["Normal", "Ischemic", "Hemorrhagic"],
                             area_vals, color=colors, height=0.55)
    for bar, v in zip(area_bars, area_vals):
        ax_area.text(min(v + 1.0, 97.0), bar.get_y() + bar.get_height() / 2,
                     f"{v:.1f}%", va="center", color="white",
                     fontsize=10, fontweight="bold")
    ax_area.set_xlim(0, 100)
    ax_area.set_xlabel("% of brain area", color="white", fontsize=10)
    brain_px_txt = f"{result.brain_area_px:,} px" if result.brain_area_px else "brain mask unavailable"
    ax_area.set_title(f"Brain area composition  ({brain_px_txt})",
                      color="white", fontsize=12, fontweight="bold", pad=6)

    # 3. Lesion overlay
    overlay = orig_np.copy().astype(np.float32)
    patches = []
    if result.brain_mask is not None:
        _draw_contours(ax_overlay, result.brain_mask.astype(np.float32),
                       "#888888", linewidth=0.8, alpha=0.6)
    if result.ischemic_mask is not None:
        overlay = _blend(overlay, result.ischemic_mask, ISCHEMIC_RGB, alpha)
        _draw_contours(ax_overlay, result.ischemic_mask, "#2196f3")
        patches.append(mpatches.Patch(
            color=[c / 255 for c in ISCHEMIC_RGB],
            label=f"Ischemic {result.ischemic_area_pct:.1f}% of brain"))
    if result.hemorrhagic_mask is not None:
        overlay = _blend(overlay, result.hemorrhagic_mask, HEMORRHAGIC_RGB, alpha)
        _draw_contours(ax_overlay, result.hemorrhagic_mask, "#ff5252")
        patches.append(mpatches.Patch(
            color=[c / 255 for c in HEMORRHAGIC_RGB],
            label=f"Hemorrhagic {result.hemorrhagic_area_pct:.1f}% of brain"))
    ax_overlay.imshow(np.clip(overlay, 0, 255).astype(np.uint8))

    if has_lesion:
        ax_overlay.set_title("Lesion Overlay (within brain)", color="white", fontsize=13, pad=8)
    else:
        ax_overlay.set_title("Lesion Overlay — No lesion detected by segmentor",
                             color="#aaaaaa", fontsize=13, pad=8)
    if patches:
        ax_overlay.legend(handles=patches, loc="lower right",
                          facecolor="#222", edgecolor="#555",
                          labelcolor="white", fontsize=10)
    ax_overlay.axis("off")

    fig.suptitle("Brain CT Stroke Analysis (3-class)",
                 color="white", fontsize=15, fontweight="bold", y=0.97)
    return fig


def _style_bar_axes(ax):
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def _blend(image: np.ndarray, mask: np.ndarray,
           color_rgb: tuple, alpha: float) -> np.ndarray:
    lesion = mask > 0.5
    color = np.array(color_rgb, dtype=np.float32)
    image[lesion] = (1 - alpha) * image[lesion] + alpha * color
    return image


def _draw_contours(ax, mask: np.ndarray, color: str,
                   linewidth: float = 1.2, alpha: float = 0.85):
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        pts = cnt[:, 0, :]
        ax.plot(np.append(pts[:, 0], pts[0, 0]),
                np.append(pts[:, 1], pts[0, 1]),
                color=color, linewidth=linewidth, alpha=alpha)


def _fig_to_numpy(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return buf.reshape(h, w, 4)[:, :, :3].copy()
