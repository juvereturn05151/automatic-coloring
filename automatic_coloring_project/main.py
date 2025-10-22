from shape_matcher import ShapeMatcher
from contour_extractor import ContourExtractor
from edge_detector import EdgeDetector
import matplotlib.pyplot as plt
import cv2

def main():
    asset_folder = "assets/"
    reference = asset_folder + "green_leg.png"
    target = asset_folder + "distorted_leg.png"

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
    edge_detector.visualize_edges(ref_img, gray, edge_normalized, num_objects)

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

    plt.figure(figsize=(6, 6))
    plt.imshow(img_boxes)
    plt.title("Detected Objects (Green Bounding Boxes)")
    plt.axis("off")
    plt.show()

    # ============================================================
    # 4. Final Auto-Colorization Results
    # ============================================================
    matcher.visualize_results(ref_img, tgt_img, colorized_match)

if __name__ == "__main__":
    main()
