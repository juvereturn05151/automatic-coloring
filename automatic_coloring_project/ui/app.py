from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import cv2

from .image_canvas import ImageCanvas
from .control_panel import ControlPanel
from .debug_viewer import DebugViewer
from .utils import save_image

# Import processing modules (from parent directory)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shape_matcher import ShapeMatcher
from contour_extractor import ContourExtractor
from edge_detector import EdgeDetector
from visualizer import draw_bounding_boxes

# Get current file's directory
CURRENT_DIR = Path(__file__).parent
ASSETS_DIR = CURRENT_DIR.parent / "assets"

DEFAULT_REFERENCE_IMAGE = ASSETS_DIR / "colored_images" / "colored_test_frame.png"
DEFAULT_TARGET_IMAGE = ASSETS_DIR / "uncolored_images" / "uncolored_test_frame.png"


class Application:
    """
    Main application class for Automatic Coloring UI.
    """

    def __init__(self):
        """Initialize the application."""
        self._root = tk.Tk()
        self._root.title("Automatic Coloring - Shape Matcher")
        self._root.resizable(False, False)

        # Image data storage
        self._reference_path = None
        self._target_path = None
        self._ref_img = None
        self._tgt_img = None
        self._result_img = None

        # Debug images
        self._binary_mask = None
        self._edge_map = None
        self._ref_with_boxes = None
        self._tgt_with_boxes = None
        self._debug_viewer = None

        # Processing components
        self._extractor = None
        self._matcher = None
        self._edge_detector = None

        # Setup UI
        self._setup_ui()
        self._setup_style()

    def _setup_style(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabelframe", background="#f0f0f0")
        style.configure("TLabelframe.Label", background="#f0f0f0")

    def _setup_ui(self):
        """Create and layout all UI components."""
        main_frame = ttk.Frame(self._root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ============================================
        # Row 1: Reference and Target Images
        # ============================================
        row1_frame = ttk.Frame(main_frame)
        row1_frame.pack(fill=tk.X, pady=(0, 10))

        self._ref_canvas = ImageCanvas(row1_frame, title="Reference (Colored)",
                                       width=400, height=400)
        self._ref_canvas.pack(side=tk.LEFT, padx=(0, 10))

        self._tgt_canvas = ImageCanvas(row1_frame, title="Target (Uncolored)",
                                       width=400, height=400)
        self._tgt_canvas.pack(side=tk.LEFT)

        # ============================================
        # Row 2: Result + Control Panel
        # ============================================
        row2_frame = ttk.Frame(main_frame)
        row2_frame.pack(fill=tk.X)

        self._result_canvas = ImageCanvas(row2_frame, title="Result (Auto-Colored)",
                                          width=400, height=400)
        self._result_canvas.pack(side=tk.LEFT, padx=(0, 10))

        callbacks = {
            'on_load_reference': self._on_load_reference,
            'on_load_target': self._on_load_target,
            'on_run': self._on_run,
            'on_save': self._on_save,
            'on_open_debug_viewer': self._on_open_debug_viewer
        }

        self._control_panel = ControlPanel(row2_frame, callbacks=callbacks)
        self._control_panel.pack(side=tk.LEFT, fill=tk.Y)

    # ===================================================
    # CALLBACK HANDLERS
    # ===================================================
    def _on_load_reference(self, file_path):
        self._reference_path = file_path
        img = cv2.imread(file_path)
        if img is not None:
            self._ref_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._update_reference_display()
            self._control_panel.set_status("Reference image loaded", "green")
        else:
            self._control_panel.set_status("Failed to load reference image", "red")

    def _on_load_target(self, file_path):
        self._target_path = file_path
        img = cv2.imread(file_path)
        if img is not None:
            self._tgt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._update_target_display()
            self._control_panel.set_status("Target image loaded", "green")
        else:
            self._control_panel.set_status("Failed to load target image", "red")

    # ===================================================
    # 🔧 FIXED VERSION OF _on_run()
    # ===================================================
    def _on_run(self):
        """Perform automatic colorization."""
        if not self._reference_path or not self._target_path:
            messagebox.showwarning("Warning", "Please load both reference and target images.")
            return

        self._control_panel.set_running(True)
        self._control_panel.set_status("Processing...", "blue")
        self._root.update()

        try:
            params = self._control_panel.get_parameters()
            min_area = params["min_area"]
            threshold = params["threshold"]

            # Convert threshold into Canny thresholds
            canny_low = max(5, threshold // 2)
            canny_high = max(30, threshold)

            # Create extractor (NEW API --- no color_clusters)
            self._extractor = ContourExtractor(
                min_area=min_area,
                canny_low=canny_low,
                canny_high=canny_high
            )

            # Create matcher + edge detector
            self._matcher = ShapeMatcher(self._extractor)
            self._edge_detector = EdgeDetector(threshold_value=threshold)

            # Process
            ref_img, tgt_img, self._result_img = self._matcher.match_and_colorize(
                self._reference_path, self._target_path
            )

            # Store for debug
            self._ref_img = ref_img
            self._tgt_img = tgt_img

            self._generate_debug_images()

            # Update displays
            self._update_reference_display()
            self._update_target_display()
            self._result_canvas.update_image(self._result_img)

            self._control_panel.set_status("Colorization complete!", "green")

        except Exception as e:
            self._control_panel.set_status(f"Error: {str(e)}", "red")
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")

        finally:
            self._control_panel.set_running(False)

    # ===================================================
    # SAVE / DEBUG HANDLERS
    # ===================================================
    def _on_save(self, file_path):
        if self._result_img is None:
            messagebox.showwarning("Warning", "No result image to save.")
            return

        if save_image(self._result_img, file_path):
            self._control_panel.set_status(f"Saved: {os.path.basename(file_path)}", "green")
        else:
            self._control_panel.set_status("Failed to save image", "red")

    def _on_open_debug_viewer(self):
        if all(img is None for img in [
            self._binary_mask, self._edge_map,
            self._ref_with_boxes, self._tgt_with_boxes
        ]):
            messagebox.showinfo("Debug Views", "Run colorization first.")
            return

        if self._debug_viewer is None or not self._debug_viewer.winfo_exists():
            self._debug_viewer = DebugViewer(self._root, on_close=self._on_debug_viewer_closed)

        self._debug_viewer.show_images(self._collect_debug_images())

    # ===================================================
    # DISPLAY UPDATES
    # ===================================================
    def _update_reference_display(self):
        if self._ref_img is not None:
            self._ref_canvas.update_image(self._ref_img)

    def _update_target_display(self):
        if self._tgt_img is not None:
            self._tgt_canvas.update_image(self._tgt_img)

    def _update_result_display(self):
        if self._result_img is not None:
            self._result_canvas.update_image(self._result_img)

    # ===================================================
    # DEBUG IMAGE GENERATION
    # ===================================================
    def _generate_debug_images(self):
        if self._extractor is None or self._ref_img is None:
            return

        try:
            # Reference boxes
            _, _, ref_contours, ref_objs = self._extractor.extract(self._reference_path)
            ref_depths = [obj.get("depth", 0) for obj in ref_objs]
            self._ref_with_boxes = draw_bounding_boxes(self._ref_img, ref_contours, ref_depths)

            # Target boxes
            _, _, tgt_contours, tgt_objs = self._extractor.extract(self._target_path)
            tgt_depths = [obj.get("depth", 0) for obj in tgt_objs]
            self._tgt_with_boxes = draw_bounding_boxes(self._tgt_img, tgt_contours, tgt_depths)

            # Edges
            if self._edge_detector:
                _, self._edge_map, _ = self._edge_detector.detect_edges(self._ref_img)

            # Binary mask
            gray = cv2.cvtColor(self._ref_img, cv2.COLOR_RGB2GRAY)
            self._binary_mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 5
            )

        except Exception as e:
            print(f"[Warning] Failed to generate debug images: {e}")

        self._update_debug_viewer_if_open()

    def _collect_debug_images(self):
        images = {}
        if self._binary_mask is not None:
            images["Binary Mask (Reference)"] = self._binary_mask
        if self._edge_map is not None:
            edge_uint8 = (self._edge_map * 255).clip(0, 255).astype("uint8")
            images["Edge Map"] = edge_uint8
        if self._ref_with_boxes is not None:
            images["Reference + Bounding Boxes"] = self._ref_with_boxes
        if self._tgt_with_boxes is not None:
            images["Target + Bounding Boxes"] = self._tgt_with_boxes
        return images

    def _update_debug_viewer_if_open(self):
        if self._debug_viewer and self._debug_viewer.winfo_exists():
            self._debug_viewer.show_images(self._collect_debug_images())

    def _on_debug_viewer_closed(self):
        self._debug_viewer = None

    # ===================================================
    # ENTRY POINT
    # ===================================================
    def run(self):
        self._root.update_idletasks()
        w = self._root.winfo_width()
        h = self._root.winfo_height()
        x = (self._root.winfo_screenwidth() // 2) - (w // 2)
        y = (self._root.winfo_screenheight() // 2) - (h // 2)
        self._root.geometry(f"+{x}+{y}")

        self._on_load_reference(str(DEFAULT_REFERENCE_IMAGE))
        self._on_load_target(str(DEFAULT_TARGET_IMAGE))

        self._root.mainloop()
