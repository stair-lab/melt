"""
Configuration file for the Sphinx documentation builder.

This file contains a selection of the most common options.
For a full list, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys
from datetime import datetime

# Path setup
sys.path.insert(0, os.path.abspath("../../src"))

# Project information
PROJECT = "MELTs"
AUTHOR = "Thu Nguyen Hoang Anh"
COPYRIGHT = f"{datetime.now().year}, {AUTHOR}"

# The full version, including alpha/beta/rc tags
RELEASE = "0.1"

# General configuration
MASTER_DOC = "index"

# Sphinx extension modules as strings, can be built-in or custom
EXTENSIONS = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx_rtd_theme",
    "sphinx.ext.doctest",
]

# List of modules to mock during autodoc generation
AUTODOC_MOCK_IMPORTS = ["pyemd"]

# Paths that contain templates
TEMPLATES_PATH = ["_templates"]

# List of patterns to ignore when looking for source files
EXCLUDE_PATTERNS = []

# Sort members alphabetically in the autodoc
AUTODOC_MEMBER_ORDER = "alphabetical"

# Options for HTML output
HTML_THEME = "sphinx_rtd_theme"

# Paths for custom static files (like style sheets)
HTML_STATIC_PATH = ["_static"]
