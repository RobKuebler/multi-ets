"""Builds the `multi_ets._holt_hot` Cython extension as part of every install.

The package metadata lives in `pyproject.toml`; this file only declares the
compiled extension (see docs/adr/0005-ship-compiled-wheels.md). The
`MULTI_ETS_EXTENSION` environment variable controls what happens:

- `optional` (default): try to compile; if the C compiler is missing or the
  compile fails, warn and install the pure-Python package, which falls back to
  the NumPy path at import time.
- `required`: a failed compile fails the build. Used by CI and the
  cibuildwheel release build, so a broken compile can't ship silently.
- `skip`: don't build the extension at all (tests the pure-Python fallback).
"""

import os
import sys
from typing import Any

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

MODE = os.environ.get("MULTI_ETS_EXTENSION", "optional")
if MODE not in {"optional", "required", "skip"}:
    raise ValueError(f"MULTI_ETS_EXTENSION must be optional/required/skip, got {MODE!r}")


class OptionalBuildExt(build_ext):
    """`build_ext` that degrades to a warning unless the extension is required."""

    def run(self) -> None:
        """Build all extensions, swallowing failures in `optional` mode."""
        try:
            super().run()
        except Exception as exc:
            self._fail(exc)

    def build_extension(self, ext: Any) -> None:
        """Build one extension, swallowing failures in `optional` mode."""
        try:
            super().build_extension(ext)
        except Exception as exc:
            self._fail(exc)

    def _fail(self, exc: Exception) -> None:
        if MODE == "required":
            raise exc
        print(
            f"WARNING: building multi_ets._holt_hot failed ({exc}); installing the "
            "pure-Python version, which uses the slower NumPy fitting path.",
            file=sys.stderr,
        )


ext_modules = []
if MODE != "skip":
    ext_modules = cythonize(
        [
            Extension(
                "multi_ets._holt_hot",
                ["src/multi_ets/_holt_hot.pyx"],
                include_dirs=[np.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
        ],
        compiler_directives={"language_level": "3"},
    )

setup(ext_modules=ext_modules, cmdclass={"build_ext": OptionalBuildExt})
