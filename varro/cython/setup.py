from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='Fast CRAM loading',
        ext_modules=cythonize("fast_cram.pyx", compiler_directives={'language_level' : "3"}),
        include_dirs=[numpy.get_include()]
    )
