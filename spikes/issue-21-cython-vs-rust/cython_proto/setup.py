import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        [Extension("holt_hot", ["holt_hot.pyx"], include_dirs=[np.get_include()])],
        compiler_directives={"language_level": "3"},
    )
)
