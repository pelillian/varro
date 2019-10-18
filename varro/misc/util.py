"""
This module contains general utility functions.
"""

import os


def make_path(dir):
    """Ensures a given path exists."""
    os.makedirs(dir, exist_ok=True)

