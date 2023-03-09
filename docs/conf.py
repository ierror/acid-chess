# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent / "..")))

from acid.conf import PROGRAM_NAME, PROGRAM_VERSION  # isort:skip

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

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_favicon = "_static/images/icon.png"
html_logo = "_static/images/icon.png"

html_context = {"default_mode": "dark"}

html_theme_options = {
    "show_toc_level": 2,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "logo": {
        "alt_text": "ACID Chess logo",
        "text": "ACID Chess",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ierror/acid-chess",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "Mastodon",
            "url": "https://chaos.social/@boerni",
            "icon": "fa-brands fa-mastodon",
        },
    ],
}
