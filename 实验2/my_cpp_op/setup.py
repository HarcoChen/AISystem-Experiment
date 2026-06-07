from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='my_op_cpp',
      ext_modules=[cpp_extension.CppExtension('my_op_cpp', ['my_op_c++.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
