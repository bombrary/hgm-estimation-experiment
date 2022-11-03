from setuptools import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpyを使う場合に必要

# 「sample」はpyxファイルごとに修正
ext1 = Extension("pfaffian_gamma1on20_sigma1_varob1", sources=["pfaffian_gamma1on20_sigma1_varob1.pyx"], include_dirs=['.', get_include()])
ext2 = Extension("pfaffian_denoms_gamma1on20_sigma1_varob1", sources=["pfaffian_denoms_gamma1on20_sigma1_varob1.pyx"], include_dirs=['.', get_include()])
ext3 = Extension("pfaffian_numer_gamma1on20_sigma1_varob1", sources=["pfaffian_numer_gamma1on20_sigma1_varob1.pyx"], include_dirs=['.', get_include()])
setup(name="pfaffian_gamma1on20_sigma1_varob1",
      ext_modules=cythonize([ext1, ext2, ext3], compiler_directives={'language_level':"3"}))
