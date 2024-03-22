# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath("../../src/lean_dojo/"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "LeanDojo"
copyright = "2023, LeanDojo Team"
author = "Kaiyu Yang"
release = "1.7.2rc"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "sphinx.ext.viewcode", "sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
html_context = {
    "display_github": True,
    "github_user": "lean-dojo",
    "github_repo": "LeanDojo",
    "github_version": "main",
}
