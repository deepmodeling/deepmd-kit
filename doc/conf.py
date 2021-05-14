# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
# import sys
import recommonmark
from recommonmark.transform import AutoStructify

def mkindex(dirname):
    dirname = dirname + "/"
    oldfindex = open(dirname + "index.md", "r")
    oldlist = oldfindex.readlines()
    oldfindex.close()

    oldnames = []
    for entry in oldlist:
        _name = entry[entry.find("(")+1 : entry.find(")")]
        oldnames.append(_name)
    
    newfindex = open(dirname + "index.md", "a")
    for root, dirs, files in os.walk(dirname, topdown=False):
        newnames = [name for name in files if name != "index.md" and name not in oldnames]
        for name in newnames:
            f = open(dirname + name, "r")
            _lines = f.readlines()
            for _headline in _lines:
                _headline = _headline.strip("#")
                headline = _headline.strip()
                if (len(headline) == 0 or headline[0] == "." or headline[0] == "="):
                    continue
                else:
                    break
            longname = "- ["+headline+"]"+"("+name+")\n"
            newfindex.write(longname)

    
    newfindex.close()


# -- Project information -----------------------------------------------------

project = 'DeePMD-kit'
copyright = '2020, Deep Potential'
author = 'Deep Potential'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = [
#     'recommonmark',
#     "sphinx_rtd_theme",
#     'myst_parser',
#     'sphinx_markdown_tables',
#     'sphinx.ext.autosummary'
# ]

mkindex("troubleshooting")
mkindex("development")

extensions = [
    "sphinx_rtd_theme",
    'myst_parser',
    'sphinx.ext.autosummary'
]

myst_heading_anchors = 4

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/custom.css']

autodoc_default_flags = ['members']
autosummary_generate = True
master_doc = 'index'
