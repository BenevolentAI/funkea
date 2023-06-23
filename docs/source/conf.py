# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import sys
from pathlib import Path

HOME = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, HOME.as_posix())

from funkea import __version__  # noqa: E402

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'funkea'
copyright = '2023-%s, Benjamin Tenmann' % datetime.date.today().year
author = 'Benjamin Tenmann'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "myst_nb",
]
autosummary_generate = True
nb_execution_mode = "off"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# make sure that image path is correct
for nb in (HOME / "examples").glob("*.ipynb"):
    notebook = nb.read_text()
    (Path.cwd() / "tutorials" / nb.name).write_text(notebook.replace("docs/source/", "../"))

readme = (HOME / "README.md").read_text()
readme = readme.replace("docs/source/", "./")
Path("markdown/README.md").write_text(readme)
