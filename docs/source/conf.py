onfiguration file for the Sphinx documentation builder.

This file contains a selection of common options. For a complete list,
refer to the Sphinx documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import datetime
import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
PROJECT_NAME = "MELTs"
AUTHOR_NAME = "Thu Nguyen Hoang Anh"
COPYRIGHT_YEAR = datetime.datetime.now().year
COPYRIGHT_TEXT = f"{COPYRIGHT_YEAR}, {AUTHOR_NAME}"

# The full version, including alpha/beta/rc tags
RELEASE_VERSION = "0.1"

# -- General configuration ---------------------------------------------------
# The master document is the root document for the documentation.
MASTER_DOC = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (e.g., 'sphinx.ext.*') or custom extensions.
EXTENSIONS = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx_rtd_theme",
    "sphinx.ext.doctest",
    # Uncomment these extensions if needed
    # "sphinx.ext.viewcode",  # To include source code in documentation
    # "sphinx.ext.napoleon",  # For Google-style and NumPy-style docstrings
]

# Mock imports to avoid errors when certain modules are not available
AUTODOC_MOCK_IMPORTS = ["pyemd"]

# Add paths that contain templates here, relative to this directory.
TEMPLATES_PATH = ["_templates"]

# Uncomment and configure the following lines if using `apidoc`
# APIDOC_MODULE_DIR = '../../src/melt/'
# APIDOC_OUTPUT_DIR = 'api'
# APIDOC_EXCLUDED_PATHS = []
# APIDOC_SEPARATE_MODULES = True

# List of patterns to ignore when looking for source files
EXCLUDE_PATTERNS = ['_build', 'Thumbs.db', '.DS_Store']

# Order of members in autodoc documentation
AUTODOC_MEMBER_ORDER = "alphabetical"

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages
HTML_THEME = "sphinx_rtd_theme"

# Add any paths that contain custom static files (e.g., style sheets) here,
# relative to this directory. These files are copied after the built-in static files.
HTML_STATIC_PATH = ["_static"]
