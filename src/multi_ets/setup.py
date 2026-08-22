"""Build script for the _holt_hot Cython extension (issue #22).

Wired in as the `[fast]` optional extra (#24) -- but `uv_build` (this
project's build backend) doesn't build Cython extensions as part of
`pip install`/`uv sync --extra fast`, so this script still has to be run
manually, same as issue #21's spike. See docs/adr/0002-optional-compiled-extension.md
and the README's "Fast (optional compiled) backend" section. Run from this
directory:

    uv pip install cython
    uv run python setup.py build_ext --inplace
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        [Extension("_holt_hot", ["_holt_hot.pyx"], include_dirs=[np.get_include()])],
        compiler_directives={"language_level": "3"},
    )
)
