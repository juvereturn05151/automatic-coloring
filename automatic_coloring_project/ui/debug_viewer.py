import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np

from .utils import cv2_to_photoimage


class DebugViewer(tk.Toplevel):
    """
    Floating window that displays debug images in a simple grid.
    Call show_images(...) with a mapping of {title: np.ndarray}.
    """

    def __init__(self, parent, on_close=None):
        super().__init__(parent)
        self.title("Debug Views")
        self.resizable(False, False)
        self._on_close = on_close

        # Keep references to PhotoImage objects to prevent garbage collection
        self._photo_refs = {}

        self._content = ttk.Frame(self, padding=10)
        self._content.pack(fill=tk.BOTH, expand=True)

        # Handle manual window close
        self.protocol("WM_DELETE_WINDOW", self._handle_close)

    def _handle_close(self):
        if self._on_close:
            self._on_close()
        self.destroy()

    def show_images(self, images):
        """
        Display the provided images.

        Args:
            images: dict[str, np.ndarray] mapping titles to image arrays (RGB or grayscale)
        """
        # Clear previous widgets
        for widget in self._content.winfo_children():
            widget.destroy()
        self._photo_refs.clear()

        if not images:
            ttk.Label(self._content, text="No debug images available").pack()
            return

        max_cols = 2
        row = col = 0

        for title, img in images.items():
            if img is None:
                continue

            frame = ttk.Frame(self._content, padding=5)
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="n")

            ttk.Label(frame, text=title, font=("Arial", 10, "bold")).pack(anchor="w")

            photo = self._to_photoimage(img)
            label = ttk.Label(frame, image=photo)
            label.pack()

            # Store reference
            self._photo_refs[title] = photo

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        self.deiconify()
        self.lift()
        self.focus()

    def _to_photoimage(self, img):
        """Convert np.ndarray (RGB or grayscale) to PhotoImage for display."""
        if img is None:
            return None

        # If float, scale to 0-255
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        # If grayscale, convert to RGB for consistent display sizing
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return cv2_to_photoimage(img, max_width=320, max_height=240)
