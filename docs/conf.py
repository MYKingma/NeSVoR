# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

info_dict: dict = dict()
with open(os.path.join(os.path.dirname(__file__), "../nesvor/version.py")) as fp:
    exec(fp.read(), info_dict)

project = "NeSVoR"
copyright = info_dict["__copyright__"]
author = info_dict["__author__"]
version = info_dict["__version__"]
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinxarg.ext",
]

# on_rtd = os.environ.get("READTHEDOCS", None) == "True"

# if not on_rtd:  # only import and set the theme if we're building docs locally
#     import sphinx_rtd_theme

#     html_theme = "sphinx_rtd_theme"
#     html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme = "furo"
html_title = "NeSVoR"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# pygment highlighting
from pygments.lexer import RegexLexer
from pygments import token
from sphinx.highlighting import lexers


class UsageLexer(RegexLexer):
    name = "nesvorusage"

    my_types = ["int", "float", "str"]

    tokens = {
        "root": [
            (r" nesvor [^\s\[]*", token.Name.Function),
            (r" command ", token.Name.Function),
            (r"-[^\s]+", token.Name.Variable),
            ("|".join(my_types), token.Keyword.Type),
            (r".", token.Text),
        ]
    }


class CommandLexer(RegexLexer):
    name = "nesvorcommand"

    tokens = {
        "root": [
            (r"nesvor [^\s\[]*", token.Name.Function),
            (r"-[^\s]+", token.Name.Variable),
            (r"\s\.{3}", token.Text),
            (r"[0-9.]+", token.Literal.Number),
            (r"[^\s\\]+", token.Literal.String),
            (r".", token.Text),
        ]
    }


lexers["nesvorusage"] = UsageLexer(startinline=True)
lexers["nesvorcommand"] = CommandLexer(startinline=True)

pygments_style = "default"  # "colorful"  # "one-dark"  #  # "github-dark"  # "sphinx"
highlight_language = "nesvorusage"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_js_files = [
    "js/custom.js",
]

# Output file base name for HTML help builder.
htmlhelp_basename = "nesvordoc"

# do not convert -- to long -
html_use_smartypants = False
smartquotes = False

autodoc_default_options = {
    "show-inheritance": True,
}
