from setuptools import setup

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
# https://setuptools.readthedocs.io/en/latest/setuptools.html

# version not in setup.cfg since read from file is a recent setuptools
# feature, and is troublesome for users with older setuptool versions
with open('VERSION') as version_file:
    VERSION = version_file.read().strip()

setup(
    version=VERSION
)
