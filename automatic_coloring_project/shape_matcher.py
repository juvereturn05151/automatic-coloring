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
    @staticmethod
    def best_rotational_match(tgt_contour, ref_contour ):
        step_deg = 10
        best_score = float('inf')
        for angle in range(0, 360, step_deg):
            M = cv2.getRotationMatrix2D((0,0), angle, 1.0)
            rotated = cv2.transform(tgt_contour, M)
            score = cv2.matchShapes(rotated, ref_contour, cv2.CONTOURS_MATCH_I1, 0.0)
            best_score = min(best_score, score)
        return best_score
    @staticmethod
    def hu_signature(cnt):
        if cnt is None or len(cnt) < 5:
            return np.zeros(7)
        hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
        # log transform greatly stabilizes comparisons
        return -np.sign(hu) * np.log1p(np.abs(hu))
    @staticmethod
    def hu_distance(tgt_contour,ref_contour):
        hu1 = ShapeMatcher.hu_signature(tgt_contour)
        hu2 = ShapeMatcher.hu_signature(ref_contour)
        dist = np.linalg.norm(hu1 - hu2)
        return dist
    @staticmethod
    def hu_similarity(cnt1, cnt2, sigma=0.3):
        d = ShapeMatcher.hu_distance(cnt1, cnt2)
        return float(np.exp(-d))

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

        # Compare each target contour against all reference contours
        for t_idx, tgt in enumerate(tgt_objs): # idx is index, tgt is object, tgt_objs is array of objects
            best_score = float('inf')
            best_color = (128, 128, 128)


            tgt_position = centroid(tgt["contour"])

            objectLocations = []
            objectColors = []

            for ref in ref_objs:
                # score = cv2.matchShapes(tgt["contour"], ref["contour"], cv2.CONTOURS_MATCH_I1, 0.0)
                score = self.best_rotational_match(tgt["contour"], ref["contour"])
                #score = self.hu_similarity(tgt["contour"],ref["contour"])
                if score <= best_score:
                    print("Best score found : ", t_idx)
                    best_score = score
                    best_color = ref["color"]
                    objectLocations.append(centroid( ref["contour"] ) )
                    objectColors.append(ref["color"])

            closestDistance = float('inf')
            bestIndex = -1
            for index, posibility in enumerate(objectLocations):
                _x1, _y1 = tgt_position
                _x2, _y2 = posibility
                thisDistance = abs(_x1 - _x2) + abs(_y1 - _y2)
                if closestDistance > thisDistance:
                    bestIndex = index
                    closestDistance = thisDistance

            best_color = objectColors[bestIndex]

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


def centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    # fallback: bbox center if degenerate
    x, y, w, h = cv2.boundingRect(contour)
    return (x + w // 2, y + h // 2)