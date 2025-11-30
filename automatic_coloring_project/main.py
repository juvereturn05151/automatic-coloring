"""
File Name:    main.py
Author(s):    Ju-ve Chankasemporn, Minjae Kyung
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.

Automatic Coloring Application - Main Entry Point
"""

from ui.app import Application


def main():
    """Launch the Automatic Coloring application."""
    app = Application()
    app.run()


if __name__ == "__main__":
    main()
