# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime
import deglib

project = 'Dynamic Exploration Graph'
current_year = datetime.date.today().year
copyright = f'{current_year}, Nico Hezel, Bruno Schilling'
author = 'Nico Hezel, Bruno Schilling'
release = deglib.__version__
version = deglib.__version__



# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', '.venv', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
}
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
