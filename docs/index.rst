pycoare
=======

**pycoare** is a Python package for calculating various **air-sea fluxes** from **bulk variables** (e.g., wind speed, temperature, humidity),
using code developed through the TOGA-COARE project :cite:`fairall_bulk_1996,fairall_coolskin_1996,fairall_integrated_1997`.

Included in this package are implementations of the **COARE v3.5 and v3.6 algorithms** that builds on the [original NOAA-PSL pycoare code](https://github.com/NOAA-PSL/COARE-algorithm).
This package makes very minor updates to the algorithm itself, instead focusing on improved code structure, packaging, documentation, and distribution by implementing an object oriented approach and utilizing modern Python tooling.
The goal of this new version is to improve usability and reproducibility, encourage collaboration, and ease maintenance.

Installation
------------

The latest stable version (currently a beta) can be downloaded using pip::

    >>> pip install pycoare

The package can also be added to projects via `uv <https://docs.astral.sh/uv/>`_::

    >>> uv add pycoare

You can install the most up-to-date version using::

    >>> pip install git+https://github.com/pycoare/coare

Versions
--------

pycoare contains two versions of the COARE algorithm: COARE v3.5 and COARE v3.6.

Version 3.5 was released in 2013, which made adjustments to the wind speed dependence of the Charnock parameter based on a large database of direct covariance stress observations (principally from a buoy) :cite:`edson_exchange_2013`.
This led to an increase in stress for wind speeds greater than about 18 m/s.
The roughness Reynolds number formulation of the scalar roughness length was tuned slightly to give the same values of `Ch` and `Ce` as Version 3.0.
The diurnal warm layer model was structured as a separate routine instead of embedded in a driver program.
COARE 3.5 was based on buoy data :cite:`edson_exchange_2013` and was compared to a large database (a total of 16,000 hours of observations) combining observations from NOAA, WHOI, and U. Miami :cite:`fairall_implementation_2011`.

Version 3.6 is slightly restructured and built around improvements in the representation of the effects of waves on fluxes. This includes improved relationships of surface roughness and whitecap fraction on wave parameters :cite:`fairall_air-sea_2022`.

Contribution
------------

I welcome any contributions - feel free to `raise an issue <https://github.com/pycoare/coare/issues>`_ or submit a `pull request <https://github.com/pycoare/coare/pulls>`_.
Take a look at the `contribution guide <contributing>`_ to get started!

Credits
-------

This Python implementation of the COARE algorithm was initially translated from MATLAB by
Byron Blomquist, Ludovic Bariteau, with support from the NOAA Physical Sciences Laboratory :cite:`ludovic_python_2021`.

The development of the COARE algorithm builds upon decades of research, for which I am extremely appreciative.
The history of the COARE algorithm and its development can be found by downloading :download:`this supplementary material <References/Fairall-2022-COARE-development.pdf>` :cite:`fairall_air-sea_2022`.

.. toctree::
    :maxdepth: 2
    :caption: Getting Started

    installation
    usage

.. toctree::
    :maxdepth: 2
    :caption: API Reference

    c35_api
    c36_api
    util_api

.. toctree::
    :maxdepth: 2
    :caption: Resources

    contributing
    references
