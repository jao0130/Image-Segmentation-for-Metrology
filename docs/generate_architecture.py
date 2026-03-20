"""
Generate system architecture diagram — Animal Metrology Pipeline.

Uses ASCII-safe labels only to avoid font/encoding issues on any OS.
Run: python docs/generate_architecture.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Colour palette ────────────────────────────────────────────────────────────
BG      = "#0f172a"   # dark navy background
C_DATA  = "#3b82f6"   # blue   — data / I/O
C_PROC  = "#10b981"   # green  — processing
C_MODEL = "#f59e0b"   # amber  — AI model
C_OUT   = "#6366f1"   # indigo — output
C_ARROW = "#94a3b8"   # slate  — arrows
C_TITLE = "#f1f5f9"   # near-white — title


def _box(ax, x, y, w, h, label, color, fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="#1e293b",
        linewidth=1.5, alpha=0.92, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color="white",
            fontweight="bold", linespacing=1.4, zorder=4)


def _arrow(ax, x1, y1, x2, y2, style="arc3,rad=0.0"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=C_ARROW,
            lw=1.8, mutation_scale=14,
            connectionstyle=style,
        ),
        zorder=2,
    )


def main():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Title
    ax.text(9, 9.45, "Animal Metrology Pipeline  -  System Architecture",
            ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_TITLE)

    # ── Row 1: Ingestion → Model inference ───────────────────────────────
    # Section label
    ax.text(0.35, 8.6, "Data + Inference", fontsize=7.5,
            color="#475569", style="italic", va="center")

    _box(ax,  1.7, 7.8, 2.8, 1.0, "COCO val2017\n(auto-download)", C_DATA)
    _box(ax,  5.2, 7.8, 2.8, 1.0,
         "filter_images.py\nSelect 2+ animals\n(COCO annotations)", C_PROC)
    _box(ax,  8.8, 7.8, 2.8, 1.0,
         "segment.py\nYOLOv8-seg\nInstance masks", C_MODEL)
    _box(ax, 12.4, 7.8, 2.8, 1.0,
         "eye_detect.py\nHoughCircles\n+ fallback", C_PROC)
    _box(ax, 16.1, 7.8, 2.8, 1.0,
         "keypoint_model/\nYOLOv8-pose\n(self-trained)", C_MODEL)

    _arrow(ax,  3.1, 7.8, 3.8, 7.8)   # COCO -> filter
    _arrow(ax,  6.6, 7.8, 7.4, 7.8)   # filter -> segment
    _arrow(ax, 10.2, 7.8, 11.0, 7.8)  # segment -> eye_detect
    _arrow(ax, 13.8, 7.8, 14.7, 7.8)  # eye_detect -> pose (upgrade)

    # "replaces" label
    ax.annotate("", xy=(13.5, 7.4), xytext=(15.3, 7.4),
                arrowprops=dict(arrowstyle="<->", color="#475569",
                                lw=1.1, linestyle="dashed"), zorder=2)
    ax.text(14.4, 7.15, "replaces after training",
            ha="center", va="center", fontsize=6.5,
            color="#475569", style="italic")

    # ── Vertical: eye_detect -> measure ──────────────────────────────────
    _arrow(ax, 12.4, 7.28, 12.4, 5.72)

    # ── Row 2: Measurement + Orchestration ───────────────────────────────
    ax.text(0.35, 5.2, "Measurement", fontsize=7.5,
            color="#475569", style="italic", va="center")

    _box(ax, 12.4, 5.0, 2.8, 1.0,
         "measure.py\nEuclidean distance\n(pixel units)", C_PROC)
    _box(ax,  8.8, 5.0, 2.8, 1.0,
         "visualize.py\nDraw masks, eyes\n& distance lines", C_PROC)
    _box(ax,  5.2, 5.0, 2.8, 1.0,
         "main.py\nPipeline\norchestrator", C_PROC)

    _arrow(ax, 11.0, 5.0, 10.2, 5.0)  # measure -> visualize
    _arrow(ax,  7.4, 5.0,  6.6, 5.0)  # visualize -> main

    # ── Row 3: Outputs ────────────────────────────────────────────────────
    ax.text(0.35, 3.1, "Outputs", fontsize=7.5,
            color="#475569", style="italic", va="center")

    _box(ax,  5.2, 2.4, 2.8, 1.0,
         "output/images/\n{id}_result.jpg", C_OUT)
    _box(ax,  8.8, 2.4, 2.8, 1.0,
         "output/\nmeasurements.csv", C_OUT)
    _box(ax, 12.4, 2.4, 2.8, 1.0,
         "docs/\narchitecture.png", C_OUT)

    _arrow(ax,  5.2, 4.5,  5.2, 2.9)   # main -> output images
    _arrow(ax,  8.8, 4.5,  8.8, 2.9)   # -> csv
    _arrow(ax, 12.4, 4.5, 12.4, 2.9)   # -> architecture diagram

    # measure -> CSV curved arrow
    _arrow(ax, 12.4, 4.5, 8.8, 2.9, style="arc3,rad=-0.25")

    # ── Legend ────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=C_DATA,  label="Data source"),
        mpatches.Patch(color=C_PROC,  label="Processing step"),
        mpatches.Patch(color=C_MODEL, label="AI model"),
        mpatches.Patch(color=C_OUT,   label="Output artifact"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              bbox_to_anchor=(0.01, 0.01), fontsize=8,
              framealpha=0.3, facecolor="#1e293b",
              edgecolor="#334155", labelcolor="white")

    # ── Save ─────────────────────────────────────────────────────────────
    out = Path("docs/architecture.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.5)
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
