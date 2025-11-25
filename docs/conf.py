# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mievformer'
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
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
    'nbsphinx',
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_mock_imports = [
    'torch', 'numpy', 'pandas', 'scanpy', 'anndata', 
    'pytorch_lightning', 'sklearn', 'matplotlib', 'seaborn', 'scipy',
    'einops', 'statsmodels', 'squidpy'
]

nbsphinx_execute = 'never'
