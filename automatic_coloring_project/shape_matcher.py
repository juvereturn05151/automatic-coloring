"""
File Name:    shape_matcher.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import cv2
import numpy as np

from contour_extractor import ContourExtractor

class ShapeMatcher:
    def __init__(self, contour_extractor=None):
        self._extractor = contour_extractor or ContourExtractor()

    def match_and_colorize(self, reference_path, target_path):
        """Extract shapes and colors"""
        ref_img, _, _, ref_objs = self._extractor.extract(reference_path)
        tgt_img, _, tgt_contours, tgt_objs = self._extractor.extract(target_path)

        colorized_match = tgt_img.copy()
        print(f"Reference objects: {len(ref_objs)} | Target objects: {len(tgt_objs)}")

        #new color & similar object dissociation variables
        amountOfBestShapes = [0] * len(tgt_objs)
        bestColors = [(0, 0, 0)] * len(tgt_objs)
        colorsRecorded = 0
        objectLocations = []

        # Compare each target contour against all reference contours
        for t_idx, tgt in enumerate(tgt_objs): # idx is index, tgt is object, tgt_objs is array of objects
            best_score = float('inf')
            best_color = (128, 128, 128)
            for ref in ref_objs:
                score = cv2.matchShapes(tgt["contour"], ref["contour"], cv2.CONTOURS_MATCH_I1, 0.0)
                if score < best_score:
                    print("Best score found : ", t_idx)
                    best_score = score
                    best_color = ref["color"]

            #region new color being referenced
            color_exists = False
            for i in range(colorsRecorded):
                if bestColors[i] == best_color:
                    color_exists = True
                    amountOfBestShapes[i] += 1
            if color_exists == False:
                bestColors[colorsRecorded] = best_color
                amountOfBestShapes[colorsRecorded] += 1
                #print("New Color: ", best_color)
                colorsRecorded += 1
            #endregion

            print(f"Target {t_idx + 1}: matched color {best_color}, score={best_score:.5f}")

            # --- Step 1. Create base mask for contour outline ---
            mask = np.zeros(tgt_img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [tgt["contour"]], -1, 255, thickness=2)

            # --- Step 2. Flood-fill interior (this fills actual shape) ---
            flood_mask = mask.copy()
            h, w = flood_mask.shape
            flood_mask_pad = np.zeros((h + 2, w + 2), np.uint8)  # required by floodFill
            flood_filled = flood_mask.copy()

            # Flood from a point inside the shape (OpenCV finds interior)
            # We can sample a seed point from the contour’s bounding box
            x, y, w_box, h_box = cv2.boundingRect(tgt["contour"])
            seed_point = (x + w_box // 2, y + h_box // 2)
            cv2.floodFill(flood_filled, flood_mask_pad, seed_point, 255)

            # --- Step 3. Merge outline + filled area ---
            full_region = cv2.bitwise_or(flood_filled, mask)

            # --- Step 4. Optional dilation for closed shapes ---
            full_region = cv2.morphologyEx(full_region, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

            # --- Step 5. Paint region color ---
            colorized_match[full_region == 255] = best_color

        #for i in range(colorsRecorded): # amountOfBestShapes:
            #print("Best shape count: ", amountOfBestShapes[i])

        return ref_img, tgt_img, colorized_match


