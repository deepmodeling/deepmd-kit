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
import subprocess
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
        newnames = [name for name in files if "index.md" not in name and name not in oldnames]
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

def classify_index_TS():
    dirname = "troubleshooting/"
    oldfindex = open(dirname + "index.md", "r")
    oldlist = oldfindex.readlines()
    oldfindex.close()

    oldnames = []
    sub_titles = []
    heads = []
    while(len(oldlist) > 0):
        entry = oldlist.pop(0)
        if (entry.find("(") >= 0):
            _name = entry[entry.find("(")+1 : entry.find(")")]
            oldnames.append(_name)
            continue
        if (entry.find("##") >= 0):
            _name = entry[entry.find("##")+3:-1]
            sub_titles.append(_name)
            continue
        entry.strip()
        if (entry != '\n'):
            heads.append(entry)
    
    newfindex = open(dirname + "index.md", "w")
    for entry in heads:
        newfindex.write(entry)
    newfindex.write('\n')
    sub_lists = [[],[]]
    for root, dirs, files in os.walk(dirname, topdown=False):
        newnames = [name for name in files if "index.md" not in name]
        for name in newnames:
            f = open(dirname + name, "r")
            _lines = f.readlines()
            f.close()
            for _headline in _lines:
                _headline = _headline.strip("#")
                headline = _headline.strip()
                if (len(headline) == 0 or headline[0] == "." or headline[0] == "="):
                    continue
                else:
                    break
            longname = "- ["+headline+"]"+"("+name+")\n"
            if ("howtoset_" in name):
                sub_lists[1].append(longname)
            else:
                sub_lists[0].append(longname)
    
    newfindex.write("## Trouble shooting\n")
    for entry in sub_lists[0]:
        newfindex.write(entry)
    newfindex.write("\n")
    newfindex.write("## Parameters setting\n")
    for entry in sub_lists[1]:
        newfindex.write(entry)
    newfindex.close()


# -- Project information -----------------------------------------------------

project = 'DeePMD-kit'
copyright = '2020, Deep Potential'
author = 'Deep Potential'

def run_doxygen(folder):
    """Run the doxygen make command in the designated folder"""

    try:
        retcode = subprocess.call("cd %s; doxygen Doxyfile" % folder, shell=True)
        if retcode < 0:
            sys.stderr.write("doxygen terminated by signal %s" % (-retcode))
    except OSError as e:
        sys.stderr.write("doxygen execution failed: %s" % e)


def generate_doxygen_xml(app):
    """Run the doxygen make commands if we're on the ReadTheDocs server"""

    read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

    if read_the_docs_build:
        run_doxygen("./")
    else:
        subprocess.call("doxygen Doxyfile", shell=True)

def setup(app):

    # Add hook for building doxygen xml when needed
    app.connect("builder-inited", generate_doxygen_xml)

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
classify_index_TS()

extensions = [
    "sphinx_rtd_theme",
    'myst_parser',
    'sphinx.ext.autosummary',
    'breathe',
    'exhale'
]

# breathe_domain_by_extension = {
#         "h" : "cpp",
# }
breathe_projects = {"myproject": "_build/xml/"}
breathe_default_project = "myproject"

exhale_args = {
   "containmentFolder":     "./API_CC",
    "rootFileName":          "api_cc.rst",
    "rootFileTitle":         "C++ API",
    "doxygenStripFromPath":  "..",
    # Suggested optional arguments
    # "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin":    "INPUT = ../source/api_cc/include/",
    # "unabridgedOrphanKinds": {"namespace"}
    # "listingExclude": [r"namespace_*"]
}

# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'cpp'

# 
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
