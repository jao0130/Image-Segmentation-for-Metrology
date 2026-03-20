"""Generate system architecture diagram as PNG."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")

BOXES = [
    (1.0, 6.5, "COCO val2017\n(auto-download)", "#4A90D9"),
    (4.0, 6.5, "filter_images.py\n篩選 2+ 動物圖片", "#5BA85E"),
    (7.0, 6.5, "segment.py\nYOLOv8-seg\n動物輪廓", "#E67E22"),
    (10.0, 6.5, "eye_detect.py\nHoughCircles\n眼睛偵測", "#9B59B6"),
    (7.0, 3.5, "measure.py\n距離計算\nEuclidean", "#E74C3C"),
    (4.0, 3.5, "visualize.py\n標注圖輸出", "#1ABC9C"),
    (1.0, 3.5, "output/images/\n*.jpg", "#95A5A6"),
    (10.0, 3.5, "output/\nmeasurements.csv", "#95A5A6"),
]

for x, y, label, color in BOXES:
    rect = mpatches.FancyBboxPatch(
        (x - 1.2, y - 0.6), 2.4, 1.2,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor="white",
        linewidth=2, alpha=0.9,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=8.5, color="white", fontweight="bold")

ARROWS = [
    ((2.2, 6.5), (2.8, 6.5)),   # COCO → filter
    ((5.2, 6.5), (5.8, 6.5)),   # filter → segment
    ((8.2, 6.5), (8.8, 6.5)),   # segment → eye_detect
    ((10.0, 5.9), (10.0, 4.1)), # eye_detect → measure
    ((8.8, 3.5), (8.2, 3.5)),   # measure → visualize
    ((5.8, 3.5), (5.2, 3.5)),   # visualize → output images
    ((10.0, 3.5), (11.2, 3.5)), # measure → CSV
]

for (x1, y1), (x2, y2) in ARROWS:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2))

ax.set_title("Animal Metrology Pipeline — System Architecture",
             fontsize=14, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig("docs/architecture.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: docs/architecture.png")
