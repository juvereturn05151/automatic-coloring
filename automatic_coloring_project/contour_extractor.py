"""
File Name:    contour_extractor.py
Author(s):    Ju-ve Chankasemporn
Rewritten:    unified edge-based segmentation for both colored and outline images.
"""

import cv2
import numpy as np


class ContourExtractor:
    """
    Extract regions from both colored and outline images using the SAME logic:

    1. Convert to grayscale
    2. Detect edges (Canny)
    3. Dilate edges to form "walls"
    4. Connected components on free space (non-wall pixels)
    5. Remove background component (touching outer border)
    6. Merge very small components into their largest neighbor
    7. For each remaining component, build:
         - "mask" (0/255)
         - "contour"
         - "color" = mean RGB (for reference images)
         - "depth" = 0
         - "source_index" = component id
    """

    def __init__(
        self,
        min_area=50,
        canny_low=50,
        canny_high=150,
        merge_rel_thresh=0.03,
        merge_abs_thresh=30,
        dilate_kernel_size=3,
    ):
        self._min_area = min_area
        self._canny_low = canny_low
        self._canny_high = canny_high
        self._merge_rel_thresh = merge_rel_thresh
        self._merge_abs_thresh = merge_abs_thresh
        self._kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)

    # ------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------
    def extract(self, image_path):
        """
        Extract regions from an image. Returns:
            img_rgb, gray, contours_list, objects_list

        objects_list is a list of dicts:
            {
                "contour": np.ndarray,
                "color": (r, g, b),
                "depth": 0,
                "source_index": label_id,
                "mask": 2D uint8 (0 or 255),
            }
        """
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[ContourExtractor] WARNING: cannot read '{image_path}'")
            return None, None, [], []

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        labels, bg_label = self._segment_by_edges(gray)
        labels = self._merge_small_regions(labels, bg_label)

        objects = self._labels_to_objects(img_rgb, gray, labels, bg_label)
        contours = [o["contour"] for o in objects]

        print(
            f"[ContourExtractor] '{image_path}': "
            f"{len(objects)} regions (bg_label={bg_label})"
        )

        return img_rgb, gray, contours, objects

    # ------------------------------------------------------------
    # Edge-based segmentation
    # ------------------------------------------------------------
    def _segment_by_edges(self, gray):
        """
        Use Canny edges as walls, then connected components on the free space.
        Returns: labels (int32 HxW), background_label
        """
        # 1. edges
        edges = cv2.Canny(gray, self._canny_low, self._canny_high)

        # 2. dilate edges to thicken walls
        walls = cv2.dilate(edges, self._kernel, iterations=1)

        # 3. free space (non-walls) -> binary 0/1
        free = (walls == 0).astype(np.uint8)

        # 4. connected components (background and regions)
        num_labels, labels = cv2.connectedComponents(free, connectivity=8)

        # 5. determine which label is background:
        #    any label that touches the border is a candidate; we pick the largest.
        h, w = labels.shape
        border_labels = np.concatenate(
            [
                labels[0, :],
                labels[h - 1, :],
                labels[:, 0],
                labels[:, w - 1],
            ]
        )
        border_labels = np.unique(border_labels)

        bg_label = 0
        max_area = -1
        for lb in border_labels:
            area = int(np.sum(labels == lb))
            if area > max_area:
                max_area = area
                bg_label = int(lb)

        return labels, bg_label

    # ------------------------------------------------------------
    # Merge small components into neighbors
    # ------------------------------------------------------------
    def _merge_small_regions(self, labels, bg_label):
        """
        Merge very small labels into their largest neighbor to avoid
        tiny star/eye/button regions that don't exist in the line art.
        """
        h, w = labels.shape
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Map label -> area
        area_map = {int(l): int(c) for l, c in zip(unique_labels, counts)}

        # total area of all non-background
        total_non_bg = sum(
            area for lb, area in area_map.items() if lb != bg_label
        )
        if total_non_bg <= 0:
            return labels

        abs_thresh = max(self._merge_abs_thresh, self._min_area)
        rel_thresh = self._merge_rel_thresh

        # Process labels from smallest to largest (excluding background)
        small_labels = [
            (lb, area)
            for lb, area in area_map.items()
            if lb != bg_label
        ]
        small_labels.sort(key=lambda x: x[1])

        kernel = np.ones((1, 1), np.uint8)
        labels_out = labels.copy()

        for lb, area in small_labels:
            if area <= 0 or lb == bg_label:
                continue

            if area > abs_thresh and area > rel_thresh * total_non_bg:
                # big enough, keep as its own region
                continue

            mask = (labels_out == lb).astype(np.uint8)
            if np.count_nonzero(mask) == 0:
                continue

            # Dilate and find neighbor labels
            dil = cv2.dilate(mask, kernel, iterations=1)
            neighbors = labels_out[dil > 0]
            neighbors = neighbors[neighbors != lb]
            neighbors = neighbors[neighbors != bg_label]

            if neighbors.size == 0:
                # nothing suitable to merge into; keep it
                continue

            # Merge into most overlapped neighbor
            vals, counts_n = np.unique(neighbors, return_counts=True)
            new_label = int(vals[np.argmax(counts_n)])

            labels_out[labels_out == lb] = new_label

        return labels_out

    # ------------------------------------------------------------
    # Build objects from labels
    # ------------------------------------------------------------
    def _labels_to_objects(self, img_rgb, gray, labels, bg_label):
        """
        Convert the integer label map into a list of region objects.
        """
        h, w = labels.shape
        objects = []

        unique_labels = np.unique(labels)
        for lb in unique_labels:
            if lb == bg_label:
                continue

            mask = (labels == lb).astype(np.uint8) * 255
            area = int(np.count_nonzero(mask))
            if area < self._min_area:
                continue

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            # take the largest contour for this label
            contour = max(contours, key=cv2.contourArea)

            # mean color in this region
            mean_color = cv2.mean(img_rgb, mask)
            color = tuple(int(v) for v in mean_color[:3])

            obj = {
                "contour": contour,
                "color": color,
                "depth": 0,
                "source_index": int(lb),
                "mask": mask,
            }
            objects.append(obj)

        return objects
