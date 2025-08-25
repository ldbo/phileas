# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime

import phileas

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "phileas"
copyright = f"{datetime.datetime.now().year}, Louis Dubois"
author = "Louis Dubois"
release = phileas.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
]
templates_path = ["_templates"]

## MyST-NB
myst_enable_extensions = [
    "colon_fence",
    "smartquotes",
    "replacements",
    "dollarmath",
]
nb_number_source_lines = True
nb_merge_streams = True
myst_heading_anchors = 6

## Autodoc
autodoc_type_aliases = {
    "DataTree": "phileas.iteration.base.DataTree",
    "DataLiteral": "phileas.iteration.base.DataLiteral",
    "PseudoDataTree": "phileas.iteration.base.PseudoDataTree",
    "PseudoDataLiteral": "phileas.iteration.base.PseudoDataLiteral",
    "Key": "phileas.iteration.base.Key",
}
autodoc_member_order = "bysource"

## Autosummary
autosummary_generate = True

## Coverage
coverage_modules = ["phileas"]
coverage_statistics_to_stdout = True

## Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

## Napoleon
napoleon_google_docstring = True

## Todo
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ldbo/phileas",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/phileas",
            "icon": "fa-custom fa-pypi",
            "type": "fontawesome",
        },
    ]
}
