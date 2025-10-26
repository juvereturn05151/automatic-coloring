import cv2
import numpy as np
import matplotlib.pyplot as plt

class ContourExtractor:
    def __init__(self, min_area=50, morph_kernel_size=5, color_clusters=4):
        self.min_area = min_area
        self.kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        self.color_clusters = color_clusters

    def preprocess(self, image_path):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return img_rgb, gray

    def is_outline_drawing(self, gray):
        pixels = gray.flatten()
        dark_ratio = np.sum(pixels < 60) / len(pixels)
        bright_ratio = np.sum(pixels >= 200) / len(pixels)
        return bright_ratio > 0.7 and dark_ratio < 0.1

    def color_segmentation(self, img_rgb):
        """Cluster colors using k-means for multi-color separation."""
        Z = img_rgb.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = self.color_clusters
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()].reshape(img_rgb.shape)
        return segmented, labels.reshape(img_rgb.shape[:2])

    def extract(self, image_path):
        img_rgb, gray = self.preprocess(image_path)
        outline_mode = self.is_outline_drawing(gray)

        if outline_mode:
            threshold_value = 250
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)
            mode = "outline (black lines)"
            mask_list = [binary]
        else:
            mode = "colored (multi-region)"
            segmented, label_map = self.color_segmentation(img_rgb)
            mask_list = []
            for k in range(self.color_clusters):
                mask = np.uint8(label_map == k) * 255
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
                mask_list.append(mask)

        # --- Contour detection ---
        objects = []
        print(f"\n[ContourExtractor] Mode: {mode}")
        for idx, mask in enumerate(mask_list):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < self.min_area:
                    continue
                region_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(region_mask, [c], -1, 255, thickness=cv2.FILLED)
                mean_color = cv2.mean(img_rgb, region_mask)
                mean_color = tuple(int(v) for v in mean_color[:3])
                objects.append({"contour": c, "color": mean_color})

        print(f"Detected {len(objects)} color regions in '{image_path}'")
        return img_rgb, gray, [obj["contour"] for obj in objects], objects

    def visualize_bounding_boxes(self, image, contours, title="Detected Objects"):
        img_copy = image.copy()

        for i, cnt in enumerate(contours, start=1):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img_copy, str(i), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

        plt.figure(figsize=(6, 6))
        plt.imshow(img_copy)
        plt.title(title)
        plt.axis("off")
        plt.show()