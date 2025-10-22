import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Helper: Extract contours and average colors
# ============================================================
def extract_contours(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 50]

    objects = []
    for c in contours:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        mean_color = cv2.mean(img_rgb, mask)
        mean_color = tuple(int(v) for v in mean_color[:3])
        objects.append({"contour": c, "color": mean_color})
    return img_rgb, gray, contours, objects


# ============================================================
# Main: Show edge detection, boxes, filled output, shape match
# ============================================================
def show_edge_detection(reference_path, target_path):
    # -------------------------------
    # 1) Read and preprocess
    # -------------------------------
    img = cv2.imread(reference_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edge = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge = cv2.convertScaleAbs(edge)
    edge = cv2.GaussianBlur(edge, (3, 3), 0)
    edge_normalized = edge.astype(np.float32) / 255.0

    # -------------------------------
    # 2) Threshold & Contour Detection
    # -------------------------------
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 50]
    num_objects = len(contours)
    print(f"Detected objects (base): {num_objects}")

    # -------------------------------
    # 3) Visualization: Original + Edge
    # -------------------------------
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original RGB Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(edge_normalized, cmap="gray")
    plt.title(f"Laplacian Edge Map\nObjects: {num_objects}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 4) Bounding Boxes
    # -------------------------------
    img_boxes = img.copy()
    for i, cnt in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img_boxes, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_boxes)
    plt.title("Detected Objects (Green Bounding Boxes)")
    plt.axis("off")
    plt.show()

    # -------------------------------
    # 5) Shape-based color transfer (position independent)
    # -------------------------------
    ref_img, ref_gray, _, ref_objs = extract_contours(reference_path)
    tgt_img, tgt_gray, tgt_contours, tgt_objs = extract_contours(target_path)

    colorized_match = tgt_img.copy()
    print(f"Reference objects: {len(ref_objs)} | Target objects: {len(tgt_objs)}")

    for t_idx, tgt in enumerate(tgt_objs):
        best_score = float('inf')
        best_color = (128, 128, 128)
        for ref in ref_objs:
            score = cv2.matchShapes(tgt["contour"], ref["contour"], cv2.CONTOURS_MATCH_I1, 0.0)
            if score < best_score:
                best_score = score
                best_color = ref["color"]
        print(f"Target {t_idx+1}: matched color {best_color}, score={best_score:.5f}")
        cv2.drawContours(colorized_match, [tgt["contour"]], -1, best_color, thickness=cv2.FILLED)

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1); plt.imshow(ref_img); plt.title("Reference (Colored)"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(tgt_img); plt.title("Target (Moved Shapes)"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(colorized_match); plt.title("Auto-Colored (Shape Matched)"); plt.axis("off")
    plt.tight_layout()
    plt.show()

    return colorized_match


# ============================================================
# Run everything
# ============================================================
if __name__ == "__main__":
    show_edge_detection("red_shirt.png", "dead_red_shirt.png")
