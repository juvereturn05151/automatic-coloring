import tkinter as tk
from tkinter import ttk

from .utils import cv2_to_photoimage, create_placeholder_image


class ImageCanvas(ttk.Frame):
    """
    A frame widget that displays an image with a title label.
    Supports updating the displayed image and maintains fixed dimensions.
    """
    
    def __init__(self, parent, title="Image", width=400, height=400, **kwargs):
        """
        Initialize ImageCanvas widget.
        
        Args:
            parent: parent Tkinter widget
            title: title text to display above the image
            width: canvas width in pixels
            height: canvas height in pixels
        """
        super().__init__(parent, **kwargs)
        
        self._width = width
        self._height = height
        self._photo_image = None  # Keep reference to prevent garbage collection
        self._current_image = None  # Store current numpy array
        
        # Title label
        self._title_label = ttk.Label(self, text=title, font=("Arial", 10, "bold"))
        self._title_label.pack(pady=(5, 2))
        
        # Canvas for image display
        self._canvas = tk.Canvas(
            self,
            width=width,
            height=height,
            highlightthickness=1,
        )
        self._canvas.pack(padx=5, pady=5)
        
        # Display placeholder initially
        self._show_placeholder()
    
    def _show_placeholder(self):
        """Display a placeholder image."""
        placeholder = create_placeholder_image(self._width, self._height)
        self.update_image(placeholder)
    
    def update_image(self, image, keep_original=True):
        """
        Update the displayed image.
        
        Args:
            image: numpy array (RGB format) to display
            keep_original: if True, store the original image for later retrieval
        """
        if image is None:
            self._show_placeholder()
            return
        
        if keep_original:
            self._current_image = image.copy()
        
        # Convert to PhotoImage
        self._photo_image = cv2_to_photoimage(image, self._width, self._height)
        
        if self._photo_image:
            # Clear canvas and display new image
            self._canvas.delete("all")
            
            # Center the image on canvas
            img_width = self._photo_image.width()
            img_height = self._photo_image.height()
            x = (self._width - img_width) // 2
            y = (self._height - img_height) // 2
            
            self._canvas.create_image(x, y, anchor=tk.NW, image=self._photo_image)
    
    def get_current_image(self):
        """
        Get the current image as numpy array.
        
        Returns:
            numpy array (RGB) or None if no image is set
        """
        return self._current_image
    
    def set_title(self, title):
        """
        Update the title label text.
        
        Args:
            title: new title string
        """
        self._title_label.config(text=title)
    
    def clear(self):
        """Clear the canvas and show placeholder."""
        self._current_image = None
        self._show_placeholder()
