from glob import glob
from os.path import (
    basename,
    splitext
)
from setuptools import (
    find_packages,
    setup
)

setup(
    name='ezaero',
    version='0.1.dev',
    license='MIT',
    description='A library for simple aerodynamic computations.',
    # TODO: long_description
    author='Pedro Arturo Morales Maries',
    author_email='part.morales@gmail.com',
    # TODO url (github repo url)
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
    keywords=[
        'aero',
        'aerodynamics',
        'aerospace',
        'engineering'
    ],
    install_requires=[
        'matplotlib>=2.0',
        'numpy'
    ],
    extras_require={
        'jupyter': ['notebook'],
        'dev': []
    },
)
