# Configuration file for the Sphinx documentation builder.

import sys
import os

sys.path.insert(0, os.path.abspath('..'))

# -- Project information

project = 'pyCOARE'
author = 'Andrew Scherer'

release = '1'
version = '0.2.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_search.extension',
]

autodoc_typehints = "none"

autodoc_member_order = "alphabetical"
toc_object_entries_show_parents = "hide"

source_suffix = {
    '.rst': 'restructuredtext'
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output ----------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"
html_title = "pyCOARE"

html_context = {
    "github_user": "pycoare",
    "github_repo": "coare",
    "github_version": "main",
    "doc_path": "docs",
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = dict(
    repository_url="https://github.com/pyCOARE/coare",
    repository_branch="main",
    html_title="pyCOARE",
    navigation_with_keys=False,
    path_to_docs="doc",
    use_edit_page_button=True,
    use_repository_button=True,
    use_issues_button=True,
    home_page_in_toc=False,
    icon_links=[],
    show_toc_level=3,
)
