"""
File Name:    main.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import cv2

from shape_matcher import ShapeMatcher
from contour_extractor import ContourExtractor
from edge_detector import EdgeDetector
from visualizer import visualize_bounding_boxes, visualize_results, visualize_edges

ASSET_FOLDER = "assets/"
COLORED_IMAGE_FOLDER = ASSET_FOLDER + "colored_images/"
UNCOLORED_IMAGE_FOLDER = ASSET_FOLDER + "uncolored_images/"

def main():
    # Change the File(s)' Name Here to Test With Other Images
    reference = COLORED_IMAGE_FOLDER + "red_shirt_eye_color.png"
    target = UNCOLORED_IMAGE_FOLDER + "red_shirt_closed_uncolored.png"

    # ============================================================
    # 1. Shape Matching and Auto-Colorization
    # ============================================================
    matcher = ShapeMatcher(ContourExtractor())
    ref_img, tgt_img, colorized_match = matcher.match_and_colorize(reference, target)

    # ============================================================
    # 2. Edge Detection and Visualization
    # ============================================================
    edge_detector = EdgeDetector()
    gray, edge_normalized, num_objects = edge_detector.detect_edges(ref_img)
    print(f"Laplacian edge objects detected: {num_objects}")
    visualize_edges(ref_img, gray, edge_normalized, num_objects)

    # ============================================================
    # 3. Bounding Boxes Visualization
    # ============================================================
    extractor = ContourExtractor()
    _, _, contours, _ = extractor.extract(reference)
    img_boxes = ref_img.copy()
    for i, cnt in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img_boxes, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

    # Reference
    ref_img, _, ref_contours, _ = extractor.extract(reference)
    visualize_bounding_boxes(ref_img, ref_contours, "Reference - Detected Objects")

    # Target
    tgt_img, _, tgt_contours, _ = extractor.extract(target)
    visualize_bounding_boxes(tgt_img, tgt_contours, "Target - Detected Objects")

    # ============================================================
    # 4. Final Auto-Colorization Results
    # ============================================================
    visualize_results(ref_img, tgt_img, colorized_match)

if __name__ == "__main__":
    main()
