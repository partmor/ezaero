ezaero
======

*(Read: easy-aero)*

.. image:: https://github.com/partmor/ezaero/raw/master/docs/source/examples/cl_distribution.png
   :align: center

This library is a free-time project. I am using it as an excuse to:

1) Learn how to properly test and package a Python library.
2) Experiment the performance of several scientific computing packages and tools (NumPy, Numba, etc.)
3) Redo *properly* (in terms of computational performance, SW best practices, and so on) a project I enjoyed a lot during my Master Thesis, back in 2014. I was curious to know how much could I improve the code performance.


My thesis covered the analysis of the aeroelastic response of an UAV in a gust scenario.

My plan is to implement the following modules in order:

+ 3D steady VLM
+ 3D then unsteady VLM
+ wing motion equation solver

If for some reason you run across this project, and find it useful or have suggestions,
don't be shy! feel free to contribute or `drop me a line <mailto:part.morales@gmail.com>`_.

Installation
------------

To install the package, clone the repo and pip install it:

.. code-block::

    $ git clone https://github.com/partmor/ezaero.git && cd ezaero
    $ pip install . # the dot means this directory

