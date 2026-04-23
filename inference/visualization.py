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
    has_lesion = (result.ischemic_area_px > 0) or (result.hemorrhagic_area_px > 0)
    ncols = 3 if has_lesion else 2

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.5), facecolor="black")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.08, wspace=0.12)

    title_rgb = CLASS_TITLE_COLORS.get(result.class_name, (200, 200, 200))
    title_norm = [c / 255 for c in title_rgb]

    # 1. Original
    ax = axes[0]
    ax.imshow(orig_np)
    ax.set_title("Original CT", color="white", fontsize=13, pad=8)
    ax.axis("off")

    # 2. Class probability bar
    ax = axes[1]
    ax.set_facecolor("#1a1a1a")
    keys = ["normal", "ischemic", "hemorrhagic"]
    probs = [result.class_probs.get(k, 0.0) for k in keys]
    colors = [_BAR_PALETTE[k] for k in keys]
    bars = ax.barh([k.capitalize() for k in keys], probs, color=colors, height=0.5)
    for bar, p in zip(bars, probs):
        ax.text(min(p + 0.01, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{p:.1%}", va="center", color="white",
                fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Probability", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_title(
        f"Result: {result.class_name.upper()}  ({result.confidence:.1%})",
        color=title_norm, fontsize=13, fontweight="bold", pad=8,
    )

    # 3. Lesion overlay
    if has_lesion:
        ax = axes[2]
        overlay = orig_np.copy().astype(np.float32)
        if result.ischemic_mask is not None:
            overlay = _blend(overlay, result.ischemic_mask, ISCHEMIC_RGB, alpha)
        if result.hemorrhagic_mask is not None:
            overlay = _blend(overlay, result.hemorrhagic_mask, HEMORRHAGIC_RGB, alpha)
        ax.imshow(np.clip(overlay, 0, 255).astype(np.uint8))

        patches = []
        if result.ischemic_mask is not None:
            _draw_contours(ax, result.ischemic_mask, "#2196f3")
            patches.append(mpatches.Patch(
                color=[c / 255 for c in ISCHEMIC_RGB],
                label=f"Ischemic {result.ischemic_area_pct:.1f}%"))
        if result.hemorrhagic_mask is not None:
            _draw_contours(ax, result.hemorrhagic_mask, "#ff5252")
            patches.append(mpatches.Patch(
                color=[c / 255 for c in HEMORRHAGIC_RGB],
                label=f"Hemorrhagic {result.hemorrhagic_area_pct:.1f}%"))
        ax.set_title("Lesion Overlay", color="white", fontsize=13, pad=8)
        if patches:
            ax.legend(handles=patches, loc="lower right",
                      facecolor="#222", edgecolor="#555",
                      labelcolor="white", fontsize=10)
        ax.axis("off")

    fig.suptitle("Brain CT Stroke Analysis (3-class)",
                 color="white", fontsize=15, fontweight="bold", y=0.97)
    return fig


def _blend(image: np.ndarray, mask: np.ndarray,
           color_rgb: tuple, alpha: float) -> np.ndarray:
    lesion = mask > 0.5
    color = np.array(color_rgb, dtype=np.float32)
    image[lesion] = (1 - alpha) * image[lesion] + alpha * color
    return image


def _draw_contours(ax, mask: np.ndarray, color: str):
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        pts = cnt[:, 0, :]
        ax.plot(np.append(pts[:, 0], pts[0, 0]),
                np.append(pts[:, 1], pts[0, 1]),
                color=color, linewidth=1.2, alpha=0.85)


def _fig_to_numpy(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return buf.reshape(h, w, 4)[:, :, :3].copy()
