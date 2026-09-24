from importlib.metadata import version as get_version

project = "numba_quaternion"
author = "Kolen Cheung"
copyright = f"2021, {author}"
version = release = get_version("numba-quaternion")

extensions = [
    "myst_parser",
    "sphinx.ext.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = f"{project} {version}"

apidoc_modules = [
    {
        "path": "../src/numba_quaternion",
        "destination": "api",
        "separate_modules": True,
        "module_first": True,
    },
]
