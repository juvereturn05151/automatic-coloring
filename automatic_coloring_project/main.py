from shape_matcher import ShapeMatcher
from contour_extractor import ContourExtractor
import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():

    asset_folder = "assets/"

    reference = asset_folder + "green_leg.png"
    target = asset_folder + "distorted_leg.png"

    matcher = ShapeMatcher(ContourExtractor())
    ref_img, tgt_img, colorized_match = matcher.match_and_colorize(reference, target)

    # ============================================================
    # Edge Detection Visualization (same as your original method)
    # ============================================================

    # Convert to grayscale
    gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)

    # Apply Laplacian to detect edges
    edge = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge = cv2.convertScaleAbs(edge)

    # Apply Gaussian blur to smooth noise
    edge = cv2.GaussianBlur(edge, (3, 3), 0)

    # Normalize for display
    edge_normalized = edge.astype(np.float32) / 255.0

    # Binary mask to detect number of edge-connected components
    _, binary = cv2.threshold(edge, 30, 255, cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(binary)
    num_objects = num_labels - 1

    print(f"Laplacian edge objects detected: {num_objects}")

    # ============================================================
    # Visualization
    # ============================================================
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(ref_img)
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

    # ============================================================
    # Bounding Boxes on Reference
    # ============================================================
    extractor = ContourExtractor()
    _, _, contours, _ = extractor.extract(reference)

    img_boxes = ref_img.copy()
    for i, cnt in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img_boxes, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_boxes)
    plt.title("Detected Objects (Green Bounding Boxes)")
    plt.axis("off")
    plt.show()

    # ============================================================
    # Final Auto-Colorization Results
    # ============================================================
    matcher.visualize_results(ref_img, tgt_img, colorized_match)


if __name__ == "__main__":
    main()
