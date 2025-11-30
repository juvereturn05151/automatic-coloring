import cv2
import numpy as np
from PIL import Image, ImageTk


def resize_image(image, max_width=400, max_height=400):
    """
    Resize image to fit within max dimensions while maintaining aspect ratio.
    
    Args:
        image: numpy array (RGB or BGR)
        max_width: maximum width in pixels
        max_height: maximum height in pixels
    
    Returns:
        Resized numpy array
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # Calculate scale factor to fit within bounds
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def cv2_to_photoimage(image, max_width=400, max_height=400):
    """
    Convert OpenCV numpy array (RGB) to Tkinter PhotoImage.
    
    Args:
        image: numpy array in RGB format
        max_width: maximum width for display
        max_height: maximum height for display
    
    Returns:
        PIL.ImageTk.PhotoImage object
    """
    if image is None:
        return None
    
    # Resize if necessary
    resized = resize_image(image, max_width, max_height)
    
    # Handle grayscale images
    if len(resized.shape) == 2:
        pil_image = Image.fromarray(resized, mode='L')
    else:
        pil_image = Image.fromarray(resized, mode='RGB')
    
    return ImageTk.PhotoImage(pil_image)


def grayscale_to_photoimage(gray_image, max_width=400, max_height=400):
    """
    Convert grayscale numpy array to Tkinter PhotoImage.
    
    Args:
        gray_image: 2D numpy array (grayscale)
        max_width: maximum width for display
        max_height: maximum height for display
    
    Returns:
        PIL.ImageTk.PhotoImage object
    """
    if gray_image is None:
        return None
    
    # Normalize if float
    if gray_image.dtype == np.float32 or gray_image.dtype == np.float64:
        gray_image = (gray_image * 255).astype(np.uint8)
    
    resized = resize_image(gray_image, max_width, max_height)
    pil_image = Image.fromarray(resized, mode='L')
    
    return ImageTk.PhotoImage(pil_image)


def save_image(image, file_path):
    """
    Save numpy array image to file.
    
    Args:
        image: numpy array in RGB format
        file_path: output file path
    
    Returns:
        True if successful, False otherwise
    """
    if image is None:
        return False
    
    try:
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, bgr_image)
        return True
    except Exception as e:
        print(f"[Error] Failed to save image: {e}")
        return False


def create_placeholder_image(width=400, height=400, text="No Image"):
    """
    Create a placeholder image with centered text.
    
    Args:
        width: image width
        height: image height
        text: text to display
    
    Returns:
        numpy array (RGB)
    """
    # Create gray background
    placeholder = np.full((height, width, 3), 200, dtype=np.uint8)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(placeholder, text, (text_x, text_y), font, font_scale, (100, 100, 100), thickness)
    
    return placeholder
