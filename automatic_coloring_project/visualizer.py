"""
File Name:    visualizer.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import cv2
import matplotlib.pyplot as plt

def visualize_bounding_boxes(image, contours, title="Detected Objects",
                             depths=None, palette=None):
    """Draw bounding boxes around detected contours."""
    img_copy = image.copy()
    if palette is None:
        palette = [
            (0, 255, 0),    # depth 0 - green
            (255, 0, 0),    # depth 1 - red
            (0, 0, 255),    # depth 2 - blue
            (255, 165, 0),  # depth 3 - orange
            (255, 255, 0),  # depth 4 - yellow
        ]
    if depths is None:
        depths = [0] * len(contours)

    for i, cnt in enumerate(contours, start=1):
        depth = depths[i - 1] if i - 1 < len(depths) else 0
        color = palette[depth % len(palette)]
        label = f"{i}" if depth == 0 else f"{i} (d{depth})"
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 1)
        cv2.putText(img_copy, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_copy)
    plt.title(title)
    plt.axis("off")
    plt.show()


def visualize_results(ref_img, tgt_img, colorized_match):
    """Additional Visualization"""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(ref_img)
    plt.title("Reference (Colored)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(tgt_img)
    plt.title("Target (Uncolored / Moved Shapes)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(colorized_match)
    plt.title("Auto-Colored (Shape Matched)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def visualize_edges(img_rgb, gray, edge_normalized, num_objects):
    """Displays the Original, Grayscale, and Laplacian Edge maps."""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original RGB Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(edge_normalized, cmap="gray")
    plt.title(f"Laplacian Edge Map (Smoothed)\nDetected: {num_objects}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
