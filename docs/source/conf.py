"""
Configuration file for the Sphinx documentation builder.

This file contains a selection of the most common options.
For a full list, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

# Add the path to your source code here.
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------

PROJECT = "MELTs"
AUTHOR = "Thu Nguyen Hoang Anh"
COPYRIGHT = f"{datetime.datetime.now().year}, {AUTHOR}"

# The version info for the project
VERSION = "0.1"  # Short version (e.g., '0.1')
RELEASE = "0.1"  # Full version (e.g., '0.1.0')

# -- General configuration ---------------------------------------------------

MASTER_DOC = "index"  # The name of the master document

# Sphinx extensions to use
EXTENSIONS = [
    "sphinx.ext.duration",  # Measure build time
    "sphinx.ext.autodoc",   # Include documentation from docstrings
    "sphinx.ext.coverage",  # Check for documentation coverage
    "sphinx.ext.doctest",   # Test embedded doctests
    "sphinx_rtd_theme",     # Read the Docs theme
]

# Mock import for autodoc
AUTODOC_MOCK_IMPORTS = ["pyemd"]

# Paths that contain templates
TEMPLATES_PATH = ["_templates"]

# Patterns to ignore when looking for source files
EXCLUDE_PATTERNS = []

# Sort members alphabetically in the autodoc
AUTODOC_MEMBER_ORDER = "alphabetical"

# Theme to use for HTML and HTML Help pages
HTML_THEME = "sphinx_rtd_theme"

# Theme options for customizing the appearance of the theme
HTML_THEME_OPTIONS = {
    # You can add theme-specific options here
}

# Paths that contain custom static files (e.g., style sheets)
HTML_STATIC_PATH = ["_static"]
