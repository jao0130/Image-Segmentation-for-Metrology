from pathlib import Path
import cv2
import numpy as np

PALETTE = [
    (0, 255, 0), (0, 0, 255), (255, 165, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 128), (0, 128, 255), (255, 128, 0),
]


def draw_results(image_path: str, animals: list, output_path: str):
    """
    Draw on image:
    - Semi-transparent colored mask per animal
    - Contour outline
    - Red dots at eye positions
    - Yellow line + label for inter-eye distance
    - Orange lines for cross-animal right-eye distances
    Saves to output_path.
    """
    img = cv2.imread(image_path)
    overlay = img.copy()

    # Draw masks
    for i, animal in enumerate(animals):
        color = PALETTE[i % len(PALETTE)]
        overlay[animal["mask"] == 1] = color
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    for i, animal in enumerate(animals):
        color = PALETTE[i % len(PALETTE)]

        # Contour
        contours, _ = cv2.findContours(
            animal["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, contours, -1, color, 2)

        # Label
        x1, y1, _, _ = animal["bbox"]
        label = f"{animal['class_name']}#{i} {animal['confidence']:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        eyes = animal.get("eyes")
        if not eyes:
            continue

        # Eye dots
        for pt in [eyes.left, eyes.right]:
            if pt:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

        # Inter-eye line
        if eyes.left and eyes.right:
            lp = (int(eyes.left[0]), int(eyes.left[1]))
            rp = (int(eyes.right[0]), int(eyes.right[1]))
            cv2.line(img, lp, rp, (0, 255, 255), 2)
            mid = ((lp[0] + rp[0]) // 2, (lp[1] + rp[1]) // 2 - 8)
            d = animal.get("inter_eye_dist", 0) or 0
            cv2.putText(img, f"{d:.1f}px", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

    # Cross-animal right-eye lines
    for i in range(len(animals)):
        for j in range(i + 1, len(animals)):
            ea = animals[i].get("eyes")
            eb = animals[j].get("eyes")
            if ea and ea.right and eb and eb.right:
                pa = (int(ea.right[0]), int(ea.right[1]))
                pb = (int(eb.right[0]), int(eb.right[1]))
                cv2.line(img, pa, pb, (0, 128, 255), 1, cv2.LINE_AA)
                mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
                d = animals[i].get(f"cross_{j}_dist", 0) or 0
                cv2.putText(img, f"{d:.1f}px", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 255), 1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)
