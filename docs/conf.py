# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Mievformer'
copyright = '2025, NicheDynamics'
author = 'NicheDynamics'

version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'nbsphinx',
]

# Exclude patterns
exclude_patterns = ['_build', '.venv', 'Thumbs.db', '.DS_Store']

# Autosummary settings
autosummary_generate = True

# MyST parser settings
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
]

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scanpy': ('https://scanpy.readthedocs.io/en/stable/', None),
    'anndata': ('https://anndata.readthedocs.io/en/stable/', None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# GitHub Pages settings
html_baseurl = 'https://kojikoji.github.io/mievformer_package/'

# GitHub context for "Edit on GitHub" links
html_context = {
    'display_github': True,
    'github_user': 'kojikoji',
    'github_repo': 'mievformer_package',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

# Theme options
html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Mock imports for dependencies that may not be installed during doc build
autodoc_mock_imports = [
    'torch', 'numpy', 'pandas', 'scanpy', 'anndata',
    'pytorch_lightning', 'sklearn', 'matplotlib', 'seaborn', 'scipy',
    'einops', 'statsmodels', 'squidpy'
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# nbsphinx settings
nbsphinx_execute = 'never'

# Source suffix
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
