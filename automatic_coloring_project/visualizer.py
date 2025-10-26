"""
File Name:    visualizer.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import cv2
import matplotlib.pyplot as plt

def visualize_bounding_boxes(image, contours, title="Detected Objects"):
    """Draw bounding boxes around detected contours."""
    img_copy = image.copy()
    for i, cnt in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img_copy, str(i), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
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
