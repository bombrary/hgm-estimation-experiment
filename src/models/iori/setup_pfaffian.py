from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpyを使う場合に必要

# 「sample」はpyxファイルごとに修正
ext = Extension("pfaffian", sources=["pfaffian.pyx"], include_dirs=['.', get_include()])
setup(name="pfaffian", ext_modules=cythonize([ext], compiler_directives={'language_level':"3"}))
