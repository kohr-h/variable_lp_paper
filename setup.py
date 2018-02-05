# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Setup script for `variable_lp`.

Installation command::

    pip install [--user] [-e] .
"""

from __future__ import print_function, absolute_import

try:
    from Cython.Build import cythonize
    CYTHON_AVAILABLE = True
except ImportError:
    def cythonize(x):
        return None

    CYTHON_AVAILABLE = False

import numpy as np
import os
from setuptools import find_packages, setup, Extension
from setuptools.command.test import test as TestCommand
import sys


root_path = os.path.dirname(__file__)


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


test_path = os.path.join(root_path, 'variable_lp', 'test')


def find_tests():
    """Discover the test files for packaging."""
    tests = []
    for path, _, filenames in os.walk(os.path.join(root_path, test_path)):
        for filename in filenames:
            basename, suffix = os.path.splitext(filename)
            if (suffix == '.py' and
                    (basename.startswith('test_') or
                     basename.endswith('_test'))):
                tests.append(os.path.join(path, filename))

    return tests


# Determine version from top-level package __init__.py file
with open(os.path.join(root_path, 'variable_lp', '__init__.py')) as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split()[-1][1:-1]
            break


# Add new Cython modules to the dictionary. Use as key the name of the .pyx
# file and add additional fields if required. For a full description of all
# possible fields, see
# https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension
#
# The following fields can be supplied:
# - sources (list of strings)
# - include_dirs (list of strings)
# - define_macros (list of 2-tuples of strings)
# - undef_macros (list of strings)
# - library_dirs (list of strings)
# - libraries (list of strings)
# - extra_objects (list of strings)
# - extra_compile_args (list of strings)
# - extra_link_args (list of strings)
# - export_symbols (list of strings)
# - depends (list of strings)
# - language (string)
# - optional (bool)

cython_modules = {
    os.path.join('variable_lp', '_cython_impl_f32.pyx'): {
        'include_dirs': [np.get_include()]},
    os.path.join('variable_lp', '_cython_impl_f64.pyx'): {
        'include_dirs': [np.get_include()]}
}


def extension_from_spec(dict_item):
    """Convert a dictionary entry into an extension object."""
    cymod, dictval = dict_item
    sources = dictval.pop('sources', [])
    include_dirs = dictval.pop('include_dirs', [])
    define_macros = dictval.pop('define_macros', [])
    undef_macros = dictval.pop('undef_macros', [])
    library_dirs = dictval.pop('library_dirs', [])
    extra_objects = dictval.pop('extra_objects', [])
    extra_compile_args = dictval.pop('extra_compile_args', [])
    extra_link_args = dictval.pop('extra_link_args', [])
    export_symbols = dictval.pop('export_symbols', [])
    depends = dictval.pop('depends', [])
    language = dictval.pop('libraries', None)
    optional = dictval.pop('optional', True)
    pymod = os.path.splitext(cymod)[0].replace(os.path.sep, '.')
    all_sources = [cymod] + list(sources)
    return Extension(name=pymod,
                     sources=all_sources,
                     include_dirs=include_dirs,
                     define_macros=define_macros,
                     undef_macros=undef_macros,
                     library_dirs=library_dirs,
                     extra_objects=extra_objects,
                     extra_compile_args=extra_compile_args,
                     extra_link_args=extra_link_args,
                     export_symbols=export_symbols,
                     depends=depends,
                     language=language,
                     optional=optional)


extensions = [extension_from_spec(item) for item in cython_modules.items()]

setup(
    name='variable-lp',

    version=version,

    description='Components for variable Lebesgue space regularization',

    url='https://github.com/kohr-h/variable_lp_paper',

    author='Holger Kohr',
    author_email='kohr@zoho.com',

    license='MPL-2.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers

    keywords='research development mathematics prototyping imaging tomography',

    packages=find_packages(),
    package_dir={'variable_lp': 'variable_lp'},
    ext_modules=cythonize(extensions),

    install_requires=['odl', 'numpy'],
    tests_require=['pytest'],
    extras_require={
        'cython': ['cython'],
        'numba': ['numba'],
        'pygpu': ['pygpu'],
    },

    cmdclass={'test': PyTest},
)
