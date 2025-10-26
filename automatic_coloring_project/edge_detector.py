"""
File Name:    edge_detector.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import cv2
import numpy as np

class EdgeDetector:
    def __init__(self, laplacian_ksize=3, gaussian_ksize=(3, 3), threshold_value=30):
        self._laplacian_ksize = laplacian_ksize
        self._gaussian_ksize = gaussian_ksize
        self._threshold_value = threshold_value

    def detect_edges(self, img_rgb):
        """
        Detects edges in an RGB image using Laplacian and Gaussian smoothing.
        Returns edge map (normalized) and number of connected components.
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Laplacian
        edge = cv2.Laplacian(gray, cv2.CV_32F, ksize=self._laplacian_ksize)
        edge = cv2.convertScaleAbs(edge)

        # Gaussian smoothing
        edge = cv2.GaussianBlur(edge, self._gaussian_ksize, 0)

        # Normalize for visualization
        edge_normalized = edge.astype(np.float32) / 255.0

        # Threshold for connected components
        _, binary = cv2.threshold(edge, self._threshold_value, 255, cv2.THRESH_BINARY)
        num_labels, _ = cv2.connectedComponents(binary)
        num_objects = num_labels - 1

        return gray, edge_normalized, num_objects


