# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import datetime
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = ['maturin']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Project information -----------------------------------------------------

project = 'ViLLMs'
author = 'Thu Nguyen Hoang Anh'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx_rtd_theme',
    'sphinx.ext.doctest'
]

autodoc_mock_imports = ["pyemd"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# apidoc_module_dir = '../../src/vieval/'
# apidoc_output_dir = 'api'
# apidoc_excluded_paths = []
# apidoc_separate_modules = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autodoc_member_order = 'alphabetical'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']