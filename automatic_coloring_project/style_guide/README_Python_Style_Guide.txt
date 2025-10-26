============================================================
🐍 PYTHON STYLE GUIDE
Author: Ju-ve Chankasemporn
Copyright (c) 2025 DigiPen Institute of Technology. 
All rights reserved.
============================================================

This document defines the Python style conventions for all
scripts and projects by Ju-ve Chankasemporn. It follows PEP 8 
principles but includes custom preferences for simplicity and 
prototyping.

------------------------------------------------------------
1. FILE HEADER
------------------------------------------------------------
Every Python source file must begin with the following header:

"""
File Name:    example_module.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology.
              All rights reserved.
"""

Place this header before all import statements.


------------------------------------------------------------
2. IMPORTS
------------------------------------------------------------
Import order:
  1. Standard library imports
  2. Third-party imports
  3. Local project imports

Example:
    import os
    import sys

    import numpy as np
    import cv2

    from my_project.utils import helper


------------------------------------------------------------
3. NAMING CONVENTIONS
------------------------------------------------------------
Modules / Files  : lowercase_with_underscores   → data_loader.py
Classes          : PascalCase                   → class PlayerController:
Functions        : lowercase_with_underscores   → def calculate_damage():
Variables        : lowercase_with_underscores   → player_health = 100
Constants        : ALL_CAPS_WITH_UNDERSCORES    → MAX_SPEED = 10
Private Members  : prefix with _                → _internal_cache


------------------------------------------------------------
4. CODE LAYOUT
------------------------------------------------------------
• Indentation: 4 spaces per indentation level (no tabs)
• Line Length: Limit lines to 79 characters
• Blank Lines: Use 2 between top-level functions/classes,
               1 between methods in a class


------------------------------------------------------------
5. WHITESPACE RULES
------------------------------------------------------------
Use a single space after commas, colons, and around operators.

Example:
    result = a + b
    my_list = [1, 2, 3]


------------------------------------------------------------
6. FUNCTIONS AND CLASSES
------------------------------------------------------------
Always use docstrings for public functions and classes.

Example:
    def compute_distance(x, y):
        """Compute Euclidean distance between two points."""
        return (x ** 2 + y ** 2) ** 0.5

Include type hints when practical:
    def add(x: int, y: int) -> int:
        return x + y


------------------------------------------------------------
7. COMMENTS
------------------------------------------------------------
Use '#' for inline or block comments.
Keep them short, clear, and up to date.

Example:
    # Initialize player position at origin
    player_pos = (0, 0)


------------------------------------------------------------
8. ERROR HANDLING
------------------------------------------------------------
Preference: Return None instead of raising exceptions.

When encountering invalid data or operations, return None and
print a warning message instead of throwing an exception.

Example:
    def load_image(path: str):
        """Load an image and return None if it cannot be loaded."""
        img = cv2.imread(path)
        if img is None:
            print(f"[Warning] Unable to load image from '{path}'")
            return None
        return img


------------------------------------------------------------
9. DEBUGGING AND OUTPUT
------------------------------------------------------------
Preference: Use print() instead of logging.

For research, prototypes, or small projects, print statements
are preferred for their simplicity.

Example:
    print("[Info] Starting color segmentation...")
    print(f"[Debug] Found {len(contours)} contours")


------------------------------------------------------------
10. VERSION CONTROL AND LICENSING
------------------------------------------------------------
All code files must include the file header (see Section 1).

Commit messages should be short and imperative:
    ✅ "Add A* pathfinding algorithm"
    ❌ "Fixed stuff"


------------------------------------------------------------
END OF STYLE GUIDE
------------------------------------------------------------
