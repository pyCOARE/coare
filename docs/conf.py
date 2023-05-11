# Configuration file for the Sphinx documentation builder.

import sys
import os

sys.path.insert(0, os.path.abspath('..'))

# -- Project information

project = 'pyCOARE'
author = 'Andrew Scherer'

release = 'a4'
version = '2023.4a4'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

source_suffix = {
    '.rst': 'restructuredtext'
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

# html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
