#!/usr/bin/env python
import os

import numpy as np
from setuptools import find_packages, setup
from setuptools.extension import Extension

if os.name == "nt":
    std_libs = []
else:
    std_libs = ["m"]

cython_extensions = [
    Extension(
        "xcube.lib",
        ["xcube/lib.pyx"],
        language="c",
        libraries=std_libs,
        include_dirs=[np.get_include()],
    ),
]

setup(
    packages=find_packages(),
    url="https://github.com/lynx-x-ray-observatory/soxs/",
    include_package_data=True,
    ext_modules=cython_extensions,
)
