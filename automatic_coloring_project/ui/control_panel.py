import tkinter as tk
from tkinter import ttk, filedialog

from pathlib import Path


class ControlPanel(ttk.Frame):
    """
    Control panel widget containing:
    - File selection buttons (Reference, Target)
    - Parameter sliders (min_area, n_clusters, threshold)
    - Action buttons (Run, Save Result)
    - Debug viewer launcher
    """
    
    def __init__(self, parent, callbacks=None, **kwargs):
        """
        Initialize ControlPanel widget.
        
        Args:
            parent: parent Tkinter widget
            callbacks: dict of callback functions:
                - 'on_load_reference': called when reference file is selected
                - 'on_load_target': called when target file is selected
                - 'on_run': called when Run button is clicked
                - 'on_save': called when Save button is clicked
                - 'on_open_debug_viewer': called when the debug viewer button is clicked
        """
        super().__init__(parent, **kwargs)
        
        self._callbacks = callbacks or {}
        
        # Variables for parameters
        self._min_area = tk.IntVar(value=10)
        self._n_clusters = tk.IntVar(value=4)
        self._threshold = tk.IntVar(value=30)
        
        # Variables for debug toggles
        self._show_binary_mask = tk.BooleanVar(value=False)
        self._show_edge_map = tk.BooleanVar(value=False)
        self._show_bounding_boxes = tk.BooleanVar(value=False)
        
        # File paths
        self._reference_path = tk.StringVar(value="")
        self._target_path = tk.StringVar(value="")
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create and layout all widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ============================================
        # Section: File Selection
        # ============================================
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Reference file
        ref_frame = ttk.Frame(file_frame)
        ref_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(ref_frame, text="Load Reference", width=15,
                   command=self._on_load_reference).pack(side=tk.LEFT)
        self._ref_label = ttk.Label(ref_frame, text="No file selected", 
                                     foreground="gray", width=30)
        self._ref_label.pack(side=tk.LEFT, padx=(10, 0))

        
        # Target file
        tgt_frame = ttk.Frame(file_frame)
        tgt_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(tgt_frame, text="Load Target", width=15,
                   command=self._on_load_target).pack(side=tk.LEFT)
        self._tgt_label = ttk.Label(tgt_frame, text="No file selected",
                                     foreground="gray", width=30)
        self._tgt_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # ============================================
        # Section: Parameters
        # ============================================
        param_frame = ttk.LabelFrame(main_frame, text="Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Min Area slider
        self._create_slider(param_frame, "Min Area:", self._min_area, 1, 100, 0)
        # N Clusters slider
        self._create_slider(param_frame, "Color Clusters:", self._n_clusters, 2, 10, 1)
        # Threshold slider
        self._create_slider(param_frame, "Edge Threshold:", self._threshold, 10, 100, 2)
        
        # ============================================
        # Section: Actions
        # ============================================
        action_frame = ttk.LabelFrame(main_frame, text="Actions", padding=10)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        btn_frame = ttk.Frame(action_frame)
        btn_frame.pack(fill=tk.X)
        
        self._run_btn = ttk.Button(btn_frame, text="Run Colorization", 
                                    command=self._on_run)
        self._run_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self._save_btn = ttk.Button(btn_frame, text="Save Result",
                                     command=self._on_save)
        self._save_btn.pack(side=tk.LEFT)
        
        # ============================================
        # Section: Debug Toggles
        # ============================================
        debug_frame = ttk.LabelFrame(main_frame, text="Debug Views", padding=10)
        debug_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._debug_btn = ttk.Button(debug_frame, text="Open Debug Viewer",
                                     command=self._on_open_debug_viewer)
        self._debug_btn.pack(fill=tk.X)

        # ============================================
        # Section: Status
        # ============================================
        status_frame = ttk.Frame(main_frame, padding=(0, 5, 0, 0))
        status_frame.pack(fill=tk.X)
        ttk.Label(status_frame, text="Status:", width=8).pack(side=tk.LEFT)
        self._status_label = ttk.Label(status_frame, text="Ready", foreground="gray")
        self._status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def _create_slider(self, parent, label, variable, min_val, max_val, row):
        """Create a labeled slider widget."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame, text=label, width=15).pack(side=tk.LEFT)
        
        slider = ttk.Scale(frame, from_=min_val, to=max_val, 
                           variable=variable, orient=tk.HORIZONTAL, length=150)
        slider.pack(side=tk.LEFT, padx=(5, 10))
        
        value_label = ttk.Label(frame, textvariable=variable, width=5)
        value_label.pack(side=tk.LEFT)
    
    def _on_load_reference(self):
        """Handle reference file selection."""
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self._reference_path.set(file_path)
            # Show truncated filename
            filename = file_path.split("/")[-1].split("\\")[-1]
            if len(filename) > 25:
                filename = filename[:22] + "..."
            self._ref_label.config(text=filename, foreground="black")
            
            if 'on_load_reference' in self._callbacks:
                self._callbacks['on_load_reference'](file_path)
    
    def _on_load_target(self):
        """Handle target file selection."""
        file_path = filedialog.askopenfilename(
            title="Select Target Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self._target_path.set(file_path)
            # Show truncated filename
            filename = file_path.split("/")[-1].split("\\")[-1]
            if len(filename) > 25:
                filename = filename[:22] + "..."
            self._tgt_label.config(text=filename, foreground="black")
            
            if 'on_load_target' in self._callbacks:
                self._callbacks['on_load_target'](file_path)
    
    def _on_run(self):
        """Handle Run button click."""
        if 'on_run' in self._callbacks:
            self._callbacks['on_run']()
    
    def _on_save(self):
        """Handle Save button click."""
        file_path = filedialog.asksaveasfilename(
            title="Save Result Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        if file_path and 'on_save' in self._callbacks:
            self._callbacks['on_save'](file_path)

    def _on_open_debug_viewer(self):
        """Handle Debug Viewer button click."""
        if 'on_open_debug_viewer' in self._callbacks:
            self._callbacks['on_open_debug_viewer']()
    
    # ============================================
    # Public API
    # ============================================
    def get_parameters(self):
        """
        Get current parameter values.
        
        Returns:
            dict with keys: min_area, n_clusters, threshold
        """
        return {
            'min_area': self._min_area.get(),
            'n_clusters': self._n_clusters.get(),
            'threshold': self._threshold.get()
        }
    
    def get_file_paths(self):
        """
        Get selected file paths.
        
        Returns:
            tuple of (reference_path, target_path)
        """
        return self._reference_path.get(), self._target_path.get()
    
    def set_status(self, message, color="black"):
        """
        Update status label.
        
        Args:
            message: status message text
            color: text color (e.g., "green", "red", "black")
        """
        self._status_label.config(text=message, foreground=color)
    
    def set_running(self, is_running):
        """
        Enable/disable controls during processing.
        
        Args:
            is_running: True to disable controls, False to enable
        """
        state = "disabled" if is_running else "normal"
        self._run_btn.config(state=state)
        self._save_btn.config(state=state)
