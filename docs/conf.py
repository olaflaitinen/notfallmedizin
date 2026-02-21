# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "notfallmedizin"
copyright = "2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov"
author = "Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov"
release = "0.1.0"
version = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sphinx = False
html_title = "notfallmedizin documentation"
html_short_title = "notfallmedizin"

napoleon_use_param = True
napoleon_google_docstring = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# Optional dependencies: mock so docs build without torch/transformers
autodoc_mock_imports = ["torch", "torchvision", "transformers", "tokenizers", "PIL", "statsmodels"]
