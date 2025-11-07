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

    def __init__(self, min_area=10, morph_kernel_size=5, color_clusters=4):
        self._min_area = min_area
        self._kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        self._color_clusters = color_clusters
        self._last_contours = []
        self._last_hierarchy = None
        self._last_indices = []

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
        contours, hierarchy = self._find_all_contours(binary)
        self._last_contours = contours
        self._last_hierarchy = hierarchy
        self._last_indices = []

        if not contours:
            print("[Warning] No contours found. Check thresholding.")
            self._last_contours = []
            self._last_hierarchy = None
            return None

        mask_list, accepted_indices = self._create_non_overlapping_masks(gray, contours, hierarchy)
        self._last_indices = accepted_indices
        if not mask_list:
            print("[Info] No masks created after filtering.")
            plt.imshow(binary, cmap="gray")
            plt.title("Debug: Binary Mask")
            plt.show()

        return mask_list, accepted_indices

    def _threshold_outline(self, gray):
        """Apply adaptive threshold and cleanup for outlines."""
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )

        # binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
        # binary = cv2.morphologyEx(
        #     binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1
        # )
        plt.imshow(binary, cmap="gray")
        plt.title("Debug: Binary Mask")
        plt.show()
        return binary

    def _find_all_contours(self, binary):
        """Return all contours from binary image."""
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return [], None

        sorted_indices = sorted(
            range(len(contours)),
            key=lambda i: cv2.contourArea(contours[i]),
        )

        sorted_contours = [contours[i] for i in sorted_indices]

        if hierarchy is not None:
            original = hierarchy[0]
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
            remap = np.full_like(original, -1)

            def _convert(idx):
                return old_to_new.get(idx, -1) if idx != -1 else -1

            for new_idx, old_idx in enumerate(sorted_indices):
                next_idx, prev_idx, child_idx, parent_idx = original[old_idx]
                remap[new_idx] = [
                    _convert(next_idx),
                    _convert(prev_idx),
                    _convert(child_idx),
                    _convert(parent_idx),
                ]

            sorted_hierarchy = remap.reshape(1, -1, 4)
        else:
            sorted_hierarchy = None

        return sorted_contours, sorted_hierarchy

    def _calculate_depth(self, idx, hierarchy):
        """Calculate hierarchical depth of a contour given its index."""
        if hierarchy is None or len(hierarchy) == 0:
            return 0

        depth = 0
        parent = hierarchy[0][idx][3]  # hierarchy[0][idx] = [next, prev, child, parent]
        while parent != -1:
            depth += 1
            parent = hierarchy[0][parent][3]
        return depth

    def _create_non_overlapping_masks(self, gray, contours, hierarchy):
        """Create filled masks while avoiding overlapping regions and excluding odd-depth contours."""
        accepted_masks = np.zeros_like(gray, dtype=np.uint8)
        mask_list = []
        accepted_indices = []

        for idx, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area < self._min_area:
                continue

            # Skip contours with odd depth (depth 1, 3, 5, ...)
            depth = self._calculate_depth(idx, hierarchy)
            if depth % 2 == 1:
                continue
            temp_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(temp_mask, [c], -1, 255, thickness=cv2.FILLED)

            mask_list.append(temp_mask)
            accepted_indices.append(idx)

            # temp_mask = np.zeros_like(gray, dtype=np.uint8)
            # cv2.drawContours(temp_mask, [c], -1, 255, thickness=cv2.FILLED)

            # overlap = cv2.bitwise_and(temp_mask, accepted_masks)
            # overlap_ratio = np.sum(overlap > 0) / np.sum(temp_mask > 0)

            # if overlap_ratio <= 1.0:
            #     mask_list.append(temp_mask)
            #     accepted_masks = cv2.bitwise_or(accepted_masks, temp_mask)
            #     accepted_indices.append(idx)

        return mask_list, accepted_indices
    
 # ------------------------------------------------------------
    # Colored Mode Processing
    # ------------------------------------------------------------
    def _process_colored_mode(self, img_rgb):
        """Detect regions in colored images using k-means clustering."""
        self._last_contours = []
        self._last_hierarchy = None
        self._last_indices = []
        segmented, label_map = self._color_segmentation(img_rgb)
        mask_list = []
        for k in range(self._color_clusters):
            mask = np.uint8(label_map == k) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
            mask_list.append(mask)
        return mask_list, None

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
            mask_data = self._process_outline_mode(gray)
        else:
            mode = "colored (multi-region)"
            mask_data = self._process_colored_mode(img_rgb)

        if mask_data is None:
            print(f"[Info] No valid regions found in '{image_path}'.")
            return None

        mask_list, contour_indices = mask_data

        if not mask_list:
            print(f"[Info] No valid regions found in '{image_path}'.")
            return None

        objects = self._extract_objects(img_rgb, gray, mask_list, contour_indices)
        print(f"\n[ContourExtractor] Mode: {mode}")
        print(f"Detected {len(objects)} color regions in '{image_path}'")

        return img_rgb, gray, [obj["contour"] for obj in objects], objects

    def _extract_objects(self, img_rgb, gray, mask_list, contour_indices=None):
        """Generate contour objects with their average colors."""
        if contour_indices is None:
            contour_indices = [None] * len(mask_list)
        objects = []
        for mask_idx, mask in enumerate(mask_list):
            source_idx = contour_indices[mask_idx] if mask_idx < len(contour_indices) else None
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
                depth = self._contour_depth(source_idx)
                objects.append({
                    "contour": c,
                    "color": color,
                    "depth": depth,
                    "source_index": source_idx,
                })
        return objects

    def _contour_depth(self, contour_idx):
        """Compute hierarchical depth from the stored contour hierarchy."""
        if (
            contour_idx is None
            or self._last_hierarchy is None
            or len(self._last_hierarchy) == 0
        ):
            return 0

        hierarchy = self._last_hierarchy[0]
        depth = 0
        parent = hierarchy[contour_idx][3]
        while parent != -1:
            depth += 1
            parent = hierarchy[parent][3]
        return depth
