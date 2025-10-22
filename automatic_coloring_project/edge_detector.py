import cv2
import numpy as np
import matplotlib.pyplot as plt

class EdgeDetector:
    def __init__(self, laplacian_ksize=3, gaussian_ksize=(3, 3), threshold_value=30):
        self.laplacian_ksize = laplacian_ksize
        self.gaussian_ksize = gaussian_ksize
        self.threshold_value = threshold_value

    def detect_edges(self, img_rgb):
        """
        Detects edges in an RGB image using Laplacian and Gaussian smoothing.
        Returns edge map (normalized) and number of connected components.
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Laplacian
        edge = cv2.Laplacian(gray, cv2.CV_32F, ksize=self.laplacian_ksize)
        edge = cv2.convertScaleAbs(edge)

        # Gaussian smoothing
        edge = cv2.GaussianBlur(edge, self.gaussian_ksize, 0)

        # Normalize for visualization
        edge_normalized = edge.astype(np.float32) / 255.0

        # Threshold for connected components
        _, binary = cv2.threshold(edge, self.threshold_value, 255, cv2.THRESH_BINARY)
        num_labels, _ = cv2.connectedComponents(binary)
        num_objects = num_labels - 1

        return gray, edge_normalized, num_objects

    def visualize_edges(self, img_rgb, gray, edge_normalized, num_objects):
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
