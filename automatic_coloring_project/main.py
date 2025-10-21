import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Visualize Laplacian Edge Detection & Count Objects
# ============================================================
def show_edge_detection(image_path):
    # Read and convert
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Compute Laplacian edges
    edge = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge = cv2.convertScaleAbs(edge)
    edge = cv2.GaussianBlur(edge, (3, 3), 0)  # smooth noisy pixels

    # Normalize for display
    edge_normalized = edge.astype(np.float32) / 255.0

    # --------------------------------------------------------
    # Count number of connected edge objects
    # --------------------------------------------------------
    _, binary = cv2.threshold(edge, 30, 255, cv2.THRESH_BINARY)

    # Connected components with stats (bounding boxes, centroids)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    num_objects = num_labels - 1  # exclude background (label 0)
    print(f"labels: {num_labels}")
    print(f"Detected objects (by connected edge regions): {num_objects}")

    # --------------------------------------------------------
    # Visualization (Original + Edge Map)
    # --------------------------------------------------------
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original RGB Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(edge_normalized, cmap="gray")
    plt.title(f"Laplacian Edge Map\nDetected Objects: {num_objects}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # New visualization: bounding boxes on original image
    # --------------------------------------------------------
    img_boxes = img.copy()
    for i in range(1, num_labels):  # skip label 0 (background)
        x, y, w, h, area = stats[i]
        if area > 10:  # skip tiny noise regions
            cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_boxes)
    plt.title("Detected Objects (Green Bounding Boxes)")
    plt.axis("off")
    plt.show()

    return edge_normalized


# ============================================================
# Run edge detection test
# ============================================================
if __name__ == "__main__":
    edge_map = show_edge_detection("red_shirt.png")
