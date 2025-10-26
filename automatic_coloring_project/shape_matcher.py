import cv2
import numpy as np
import matplotlib.pyplot as plt
from contour_extractor import ContourExtractor

class ShapeMatcher:
    def __init__(self, contour_extractor=None):
        self.extractor = contour_extractor or ContourExtractor()

    def match_and_colorize(self, reference_path, target_path):
        # Extract shapes and colors
        ref_img, _, _, ref_objs = self.extractor.extract(reference_path)
        tgt_img, _, tgt_contours, tgt_objs = self.extractor.extract(target_path)

        colorized_match = tgt_img.copy()
        print(f"Reference objects: {len(ref_objs)} | Target objects: {len(tgt_objs)}")

        # Compare each target contour against all reference contours
        for t_idx, tgt in enumerate(tgt_objs):
            best_score = float('inf')
            best_color = (128, 128, 128)
            for ref in ref_objs:
                score = cv2.matchShapes(tgt["contour"], ref["contour"], cv2.CONTOURS_MATCH_I1, 0.0)
                if score < best_score:
                    best_score = score
                    best_color = ref["color"]

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

        return ref_img, tgt_img, colorized_match

    # Optional visualization
    def visualize_results(self, ref_img, tgt_img, colorized_match):
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
