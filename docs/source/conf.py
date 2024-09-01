"""
Configuration file for the Sphinx documentation builder.
This file contains a selection of the most common options.
For a full list, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import datetime
import os
import sys

# Add the path to the source code
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------

PROJECT = "MELTs"
AUTHOR = "Thu Nguyen Hoang Anh"
COPYRIGHT = f"{datetime.datetime.now().year}, {AUTHOR}"

# The full version, including alpha/beta/rc tags
RELEASE = "0.1"

# -- General configuration ---------------------------------------------------

MASTER_DOC = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or custom ones.
EXTENSIONS = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx_rtd_theme",
    "sphinx.ext.doctest",
]

# Mock import paths for autodoc
AUTODOC_MOCK_IMPORTS = ["pyemd"]

# Add any paths that contain templates here, relative to this directory.
TEMPLATES_PATH = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
EXCLUDE_PATTERNS = []

# Order of members in documentation
AUTODOC_MEMBER_ORDER = "alphabetical"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of built-in themes.
HTML_THEME = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. These files are copied after the built-in static files,
# so a file named "default.css" will overwrite the built-in "default.css".
HTML_STATIC_PATH = ["_static"]
