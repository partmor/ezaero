import sys
from glob import glob
from os.path import (
    basename,
    splitext
)
from setuptools import (
    find_packages,
    setup
)
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import shlex
        # import here, because outside the eggs aren't loaded
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


NAME = 'ezaero'
DESCRIPTION = 'Aerodynamics in Python.'
URL = 'https://github.com/partmor/ezaero'
EMAIL = 'part.morales@gmail.com'
AUTHOR = 'Pedro Arturo Morales Maries'
REQUIRES_PYTHON = '>=3.5'
VERSION = '0.1.dev0'

with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name=NAME,
    version=VERSION,
    license='MIT',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # TODO
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
    ],
    python_requires=REQUIRES_PYTHON,
    install_requires=[
        'matplotlib>=2.0',
        'numpy'
    ],
    tests_require=[
        'pytest'
    ],
    extras_require={
        'jupyter': ['notebook'],
        'dev': []
    },
    cmdclass={'test': PyTest}
)
