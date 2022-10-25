from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include 

ext = Extension("pfaffian_1on20_1_1_5", sources=["pfaffian_1on20_1_1_5.pyx"], include_dirs=['.', get_include()])
setup(name="pfaffian_1on20_1_1_5", ext_modules=cythonize([ext], compiler_directives={'language_level':"3"}))
