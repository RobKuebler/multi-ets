"""Build script for the _holt_hot Cython extension (issue #22).

Not wired into the package's real build yet (that's issue #24's call on
required-vs-optional-dependency + CI wheel building); mirrors the issue #21
spike's setup.py. Run from this directory:

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
