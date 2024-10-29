pycoare
=======

**pycoare** is a Python package for calculating various air-sea fluxes from bulk variables,
using code developed through the TOGA-COARE project.

Installation
------------

The latest stable version (currently a beta) can be downloaded using Pip::

    >>> pip install pycoare

You can install the most up-to-date version using::

    >>> pip install git+https://github.com/pyCOARE/coare

Credits
-------

This version of the Python implementation of the COARE algorithm was `initially translated from MATLAB by Byron Blomquist and Ludovic Bariteau <https://github.com/NOAA-PSL/COARE-algorithm>`_.
For more information on the people and publications that developed the COARE algorithm, see the references at the link below.

.. toctree::
    :maxdepth: 2
    :caption: Getting Started

    usage
    References <https://github.com/pyCOARE/coare/tree/main/docs/References>

.. toctree::
    :maxdepth: 2
    :caption: API Reference

    c35_api
    util_api
