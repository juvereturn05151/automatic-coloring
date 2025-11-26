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
    
    Layout (2-row structure):
        Row 1: [Reference Image] [Target Image]
        Row 2: [Result Image]    [Control Panel]
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
        # Main container
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
        # Row 2: Result Image and Control Panel
        # ============================================
        row2_frame = ttk.Frame(main_frame)
        row2_frame.pack(fill=tk.X)
        
        self._result_canvas = ImageCanvas(row2_frame, title="Result (Auto-Colored)",
                                           width=400, height=400)
        self._result_canvas.pack(side=tk.LEFT, padx=(0, 10))
        
        # Control Panel with callbacks
        callbacks = {
            'on_load_reference': self._on_load_reference,
            'on_load_target': self._on_load_target,
            'on_run': self._on_run,
            'on_save': self._on_save,
            'on_open_debug_viewer': self._on_open_debug_viewer
        }
        
        self._control_panel = ControlPanel(row2_frame, callbacks=callbacks)
        self._control_panel.pack(side=tk.LEFT, fill=tk.Y)
    
    # ============================================
    # Callback Handlers
    # ============================================
    def _on_load_reference(self, file_path):
        """Handle reference image loading."""
        self._reference_path = file_path
        img = cv2.imread(file_path)
        if img is not None:
            self._ref_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._update_reference_display()
            self._control_panel.set_status("Reference image loaded", "green")
        else:
            self._control_panel.set_status("Failed to load reference image", "red")
    
    def _on_load_target(self, file_path):
        """Handle target image loading."""
        self._target_path = file_path
        img = cv2.imread(file_path)
        if img is not None:
            self._tgt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._update_target_display()
            self._control_panel.set_status("Target image loaded", "green")
        else:
            self._control_panel.set_status("Failed to load target image", "red")
    
    def _on_run(self):
        """Handle Run button click - perform colorization."""
        if not self._reference_path or not self._target_path:
            messagebox.showwarning("Warning", "Please load both reference and target images.")
            return
        
        self._control_panel.set_running(True)
        self._control_panel.set_status("Processing...", "blue")
        self._root.update()
        
        try:
            # Get parameters
            params = self._control_panel.get_parameters()
            
            # Create extractor with current parameters
            self._extractor = ContourExtractor(
                min_area=params['min_area'],
                color_clusters=params['n_clusters']
            )
            
            # Create matcher and edge detector
            self._matcher = ShapeMatcher(self._extractor)
            self._edge_detector = EdgeDetector(threshold_value=params['threshold'])
            
            # Run colorization
            ref_img, tgt_img, self._result_img = self._matcher.match_and_colorize(
                self._reference_path, self._target_path
            )
            
            # Store images for debug views
            self._ref_img = ref_img
            self._tgt_img = tgt_img
            
            # Generate debug images
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
    
    def _on_save(self, file_path):
        """Handle Save button click."""
        if self._result_img is None:
            messagebox.showwarning("Warning", "No result image to save. Run colorization first.")
            return
        
        if save_image(self._result_img, file_path):
            self._control_panel.set_status(f"Saved: {file_path.split('/')[-1]}", "green")
        else:
            self._control_panel.set_status("Failed to save image", "red")

    def _on_open_debug_viewer(self):
        """Open or refresh the floating debug viewer window."""
        if all(img is None for img in [self._binary_mask, self._edge_map,
                                       self._ref_with_boxes, self._tgt_with_boxes]):
            messagebox.showinfo("Debug Views", "No debug images yet. Run colorization first.")
            return

        if self._debug_viewer is None or not self._debug_viewer.winfo_exists():
            self._debug_viewer = DebugViewer(self._root, on_close=self._on_debug_viewer_closed)

        self._debug_viewer.show_images(self._collect_debug_images())
    
    # ============================================
    # Image Display Updates
    # ============================================
    def _update_reference_display(self):
        """Always show the original reference image on the main canvas."""
        if self._ref_img is not None:
            self._ref_canvas.set_title("Reference (Colored)")
            self._ref_canvas.update_image(self._ref_img)
    
    def _update_target_display(self):
        """Always show the original target image on the main canvas."""
        if self._tgt_img is not None:
            self._tgt_canvas.set_title("Target (Uncolored)")
            self._tgt_canvas.update_image(self._tgt_img)
    
    def _update_result_display(self):
        """Show the auto-colored result on the main canvas."""
        if self._result_img is not None:
            self._result_canvas.set_title("Result (Auto-Colored)")
            self._result_canvas.update_image(self._result_img)
    
    def _generate_debug_images(self):
        """Generate debug visualization images."""
        if self._extractor is None or self._ref_img is None:
            return
        
        try:
            # Extract contours for bounding boxes
            _, _, ref_contours, ref_objs = self._extractor.extract(self._reference_path)
            ref_depths = [obj.get("depth", 0) for obj in ref_objs]
            self._ref_with_boxes = draw_bounding_boxes(self._ref_img, ref_contours, ref_depths)
            
            # Target bounding boxes
            _, _, tgt_contours, tgt_objs = self._extractor.extract(self._target_path)
            tgt_depths = [obj.get("depth", 0) for obj in tgt_objs]
            self._tgt_with_boxes = draw_bounding_boxes(self._tgt_img, tgt_contours, tgt_depths)
            
            # Edge detection
            if self._edge_detector and self._ref_img is not None:
                _, self._edge_map, _ = self._edge_detector.detect_edges(self._ref_img)
            
            # Binary mask (from extractor's last processing)
            # We'll get this from a dedicated method or recreate it
            gray = cv2.cvtColor(self._ref_img, cv2.COLOR_RGB2GRAY)
            self._binary_mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 5
            )
            
        except Exception as e:
            print(f"[Warning] Failed to generate debug images: {e}")

        # Refresh debug viewer if it's already open
        self._update_debug_viewer_if_open()

    def _collect_debug_images(self):
        """Gather available debug images into a labeled dict."""
        images = {}
        if self._binary_mask is not None:
            images["Binary Mask (Reference)"] = self._binary_mask
        if self._edge_map is not None:
            # Convert edge map (float 0-1) to display-friendly uint8
            edge_uint8 = (self._edge_map * 255.0).clip(0, 255).astype('uint8')
            images["Edge Map"] = edge_uint8
        if self._ref_with_boxes is not None:
            images["Reference + Bounding Boxes"] = self._ref_with_boxes
        if self._tgt_with_boxes is not None:
            images["Target + Bounding Boxes"] = self._tgt_with_boxes
        return images

    def _update_debug_viewer_if_open(self):
        """Update debug viewer window when it's already open."""
        if self._debug_viewer and self._debug_viewer.winfo_exists():
            self._debug_viewer.show_images(self._collect_debug_images())

    def _on_debug_viewer_closed(self):
        """Reset reference when the debug viewer window is closed."""
        self._debug_viewer = None
    
    # ============================================
    # Public API
    # ============================================
    def run(self):
        """Start the application main loop."""
        # Center window on screen
        self._root.update_idletasks()
        width = self._root.winfo_width()
        height = self._root.winfo_height()
        x = (self._root.winfo_screenwidth() // 2) - (width // 2)
        y = (self._root.winfo_screenheight() // 2) - (height // 2)
        self._root.geometry(f"+{x}+{y}")

        self._on_load_reference(str(DEFAULT_REFERENCE_IMAGE))
        self._on_load_target(str(DEFAULT_TARGET_IMAGE))
        
        self._root.mainloop()
