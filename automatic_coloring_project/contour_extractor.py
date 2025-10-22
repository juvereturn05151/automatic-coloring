import cv2
import numpy as np

class ContourExtractor:
    def __init__(self, min_area=50, morph_kernel_size=5):
        self.min_area = min_area
        self.kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)

    def preprocess(self, image_path):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return img_rgb, gray

    def extract(self, image_path):
        img_rgb, gray = self.preprocess(image_path)

        # Binary mask
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)

        # Contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > self.min_area]

        # Extract mean colors per contour
        objects = []
        for c in contours:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
            mean_color = cv2.mean(img_rgb, mask)
            mean_color = tuple(int(v) for v in mean_color[:3])
            objects.append({"contour": c, "color": mean_color})

        return img_rgb, gray, contours, objects
