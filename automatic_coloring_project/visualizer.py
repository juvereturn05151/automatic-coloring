"""
File Name:    visualizer.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.

Visualization utilities for Automatic Coloring Application.
These functions return numpy arrays for display in Tkinter UI.
"""

import cv2
import numpy as np


def draw_bounding_boxes(image, contours, depths=None, palette=None):
    """
    Draw bounding boxes around detected contours and return the result.
    
    Args:
        image: numpy array (RGB)
        contours: list of contours
        depths: list of depth values for each contour (optional)
        palette: list of RGB color tuples for different depths (optional)
    
    Returns:
        numpy array (RGB) with bounding boxes drawn
    """
    img_copy = image.copy()
    
    if palette is None:
        palette = [
            (0, 255, 0),    # depth 0 - green
            (255, 0, 0),    # depth 1 - red
            (0, 0, 255),    # depth 2 - blue
            (255, 165, 0),  # depth 3 - orange
            (255, 255, 0),  # depth 4 - yellow
        ]
    
    if depths is None:
        depths = [0] * len(contours)
    
    for i, cnt in enumerate(contours, start=1):
        depth = depths[i - 1] if i - 1 < len(depths) else 0
        color = palette[depth % len(palette)]
        label = f"{i}" if depth == 0 else f"{i} (d{depth})"
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 1)
        cv2.putText(img_copy, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1)
    
    return img_copy


def create_comparison_image(ref_img, tgt_img, colorized_match):
    """
    Create a side-by-side comparison image of reference, target, and result.
    
    Args:
        ref_img: reference image (RGB numpy array)
        tgt_img: target image (RGB numpy array)
        colorized_match: colorized result image (RGB numpy array)
    
    Returns:
        numpy array (RGB) with three images side by side
    """
    # Ensure all images have the same height
    h1, w1 = ref_img.shape[:2]
    h2, w2 = tgt_img.shape[:2]
    h3, w3 = colorized_match.shape[:2]
    
    max_h = max(h1, h2, h3)
    
    # Resize to match height if needed
    def resize_to_height(img, target_h):
        h, w = img.shape[:2]
        if h != target_h:
            scale = target_h / h
            new_w = int(w * scale)
            return cv2.resize(img, (new_w, target_h))
        return img
    
    ref_resized = resize_to_height(ref_img, max_h)
    tgt_resized = resize_to_height(tgt_img, max_h)
    result_resized = resize_to_height(colorized_match, max_h)
    
    # Concatenate horizontally
    combined = np.hstack([ref_resized, tgt_resized, result_resized])
    
    return combined


def create_edge_comparison_image(img_rgb, gray, edge_normalized, num_objects):
    """
    Create a comparison image showing original, grayscale, and edge map.
    
    Args:
        img_rgb: original RGB image
        gray: grayscale image
        edge_normalized: normalized edge map (float32, 0-1)
        num_objects: number of detected objects
    
    Returns:
        numpy array (RGB) with comparison visualization
    """
    h, w = img_rgb.shape[:2]
    
    # Convert grayscale to RGB
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # Convert edge map to RGB (scale to 0-255)
    edge_uint8 = (edge_normalized * 255).astype(np.uint8)
    edge_rgb = cv2.cvtColor(edge_uint8, cv2.COLOR_GRAY2RGB)
    
    # Add text label to edge map
    cv2.putText(edge_rgb, f"Detected: {num_objects}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Concatenate horizontally
    combined = np.hstack([img_rgb, gray_rgb, edge_rgb])
    
    return combined


# ============================================
# Legacy functions (for backward compatibility)
# These can be removed once all code is migrated to UI
# ============================================

def visualize_bounding_boxes(image, contours, title="Detected Objects",
                             depths=None, palette=None):
    """
    Legacy function - draws and displays bounding boxes using matplotlib.
    Deprecated: Use draw_bounding_boxes() instead and display in UI.
    """
    import matplotlib.pyplot as plt
    
    img_with_boxes = draw_bounding_boxes(image, contours, depths, palette)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_with_boxes)
    plt.title(title)
    plt.axis("off")
    plt.show()


def visualize_results(ref_img, tgt_img, colorized_match):
    """
    Legacy function - displays comparison using matplotlib.
    Deprecated: Use create_comparison_image() instead and display in UI.
    """
    import matplotlib.pyplot as plt
    
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


def visualize_edges(img_rgb, gray, edge_normalized, num_objects):
    """
    Legacy function - displays edge visualization using matplotlib.
    Deprecated: Use create_edge_comparison_image() instead and display in UI.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original RGB Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(edge_normalized, cmap="gray")
    plt.title(f"Laplacian Edge Map (Smoothed)\nDetected: {num_objects}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
