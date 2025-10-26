"""
File Name:    contour_extractor.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

class ContourExtractor:
    """Extract contours and regions from images based on color or outlines."""

    def __init__(self, min_area=50, morph_kernel_size=5, color_clusters=4):
        self._min_area = min_area
        self._kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        self._color_clusters = color_clusters

    # ------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------
    def preprocess(self, image_path):
        """Load image from path and convert it to RGB and grayscale."""
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Warning] Could not read image: {image_path}")
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return img_rgb, gray

    def is_outline_drawing(self, gray):
        """Return True if image is outline drawing (bright bg, dark lines)."""
        pixels = gray.flatten()
        dark_ratio = np.sum(pixels < 60) / len(pixels)
        bright_ratio = np.sum(pixels >= 200) / len(pixels)
        return bright_ratio > 0.7 and dark_ratio < 0.1

    # ------------------------------------------------------------
    # Outline Mode Processing
    # ------------------------------------------------------------
    def _process_outline_mode(self, gray):
        """Detect contours from an outline (black-line) drawing."""
        binary = self._threshold_outline(gray)
        contours = self._find_all_contours(binary)

        if not contours:
            print("[Warning] No contours found. Check thresholding.")
            return None

        mask_list = self._create_non_overlapping_masks(gray, contours)
        if not mask_list:
            print("[Info] No masks created after filtering.")
            plt.imshow(binary, cmap="gray")
            plt.title("Debug: Binary Mask")
            plt.show()

        return mask_list

    def _threshold_outline(self, gray):
        """Apply adaptive threshold and cleanup for outlines."""
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 5
        )
        binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1
        )
        return binary

    def _find_all_contours(self, binary):
        """Return all contours from binary image."""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        return sorted(contours, key=cv2.contourArea)

    def _create_non_overlapping_masks(self, gray, contours):
        """Create filled masks while avoiding overlapping regions."""
        accepted_masks = np.zeros_like(gray, dtype=np.uint8)
        mask_list = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < self._min_area:
                continue

            temp_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(temp_mask, [c], -1, 255, thickness=cv2.FILLED)

            overlap = cv2.bitwise_and(temp_mask, accepted_masks)
            overlap_ratio = np.sum(overlap > 0) / np.sum(temp_mask > 0)

            if overlap_ratio < 0.2:
                mask_list.append(temp_mask)
                accepted_masks = cv2.bitwise_or(accepted_masks, temp_mask)
        return mask_list

    # ------------------------------------------------------------
    # Colored Mode Processing
    # ------------------------------------------------------------
    def _process_colored_mode(self, img_rgb):
        """Detect regions in colored images using k-means clustering."""
        segmented, label_map = self._color_segmentation(img_rgb)
        mask_list = []
        for k in range(self._color_clusters):
            mask = np.uint8(label_map == k) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
            mask_list.append(mask)
        return mask_list

    def _color_segmentation(self, img_rgb):
        """Cluster colors using k-means for multi-color separation."""
        Z = np.float32(img_rgb.reshape((-1, 3)))
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            10, 1.0
        )
        _, labels, centers = cv2.kmeans(
            Z, self._color_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()].reshape(img_rgb.shape)
        return segmented, labels.reshape(img_rgb.shape[:2])

    # ------------------------------------------------------------
    # Contour Extraction Entry Point
    # ------------------------------------------------------------
    def extract(self, image_path):
        """Extract contours and regions based on image type."""
        result = self.preprocess(image_path)
        if result is None:
            return None
        img_rgb, gray = result
        outline_mode = self.is_outline_drawing(gray)

        if outline_mode:
            mode = "outline (black lines)"
            mask_list = self._process_outline_mode(gray)
        else:
            mode = "colored (multi-region)"
            mask_list = self._process_colored_mode(img_rgb)

        if not mask_list:
            print(f"[Info] No valid regions found in '{image_path}'.")
            return None

        objects = self._extract_objects(img_rgb, gray, mask_list)
        print(f"\n[ContourExtractor] Mode: {mode}")
        print(f"Detected {len(objects)} color regions in '{image_path}'")

        return img_rgb, gray, [obj["contour"] for obj in objects], objects

    def _extract_objects(self, img_rgb, gray, mask_list):
        """Generate contour objects with their average colors."""
        objects = []
        for mask in mask_list:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                if cv2.contourArea(c) < self._min_area:
                    continue
                region_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(region_mask, [c], -1, 255, thickness=cv2.FILLED)
                mean_color = cv2.mean(img_rgb, region_mask)
                color = tuple(int(v) for v in mean_color[:3])
                objects.append({"contour": c, "color": color})
        return objects
