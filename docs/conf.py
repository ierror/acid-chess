# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent / ".." / "src")))

from acid import PROGRAM_NAME, PROGRAM_VERSION  # isort:skip

project = PROGRAM_NAME
copyright = "2023, Bernhard Janetzki"
author = "Bernhard Janetzki"
release = PROGRAM_VERSION

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

html_theme_options = {
    "github_user": "ierror",
    "github_repo": "acid-chess",
    "fixed_sidebar": True,
    "github_banner": False,
    "github_button": True,
    "code_font_family": 'SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",Courier,monospace',
    "code_font_size": "12px",
}
