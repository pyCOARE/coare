[![Tests](https://github.com/pycoare/coare/actions/workflows/tests.yml/badge.svg)](https://github.com/pycoare/coare/actions/workflows/tests.yml)
[![License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](ttps://github.com/pycoare/coare/blob/master/LICENSE.txt)
[![Code coverage](https://codecov.io/gh/pycoare/coare/branch/main/graph/badge.svg)](https://app.codecov.io/gh/pycoare/coare)
[![Docs](https://readthedocs.org/projects/pycoare/badge/?version=latest)](https://pycoare.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/pycoare?style=plastic)](https://pypi.org/project/pycoare/)

# pycoare

**pycoare** is a Python package for calculating various **air-sea fluxes** from **bulk variables** (e.g., wind speed, temperature, humidity),
using the COARE algorithms developed through the TOGA-COARE project ([Fairall et al., 1996a](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/95JC03190), [Fairall et al., 1996b](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/95JC03205), [Fairall et al., 1997](http://journals.ametsoc.org/doi/10.1175/1520-0426(1997)014%3C0338:ISMOTM%3E2.0.CO;2)).

Included in this package are implementations of the **COARE v3.5 and v3.6 algorithms** that builds on the [original NOAA-PSL pycoare code](https://github.com/NOAA-PSL/COARE-algorithm). This package makes very minor updates to the algorithm itself, instead focusing on improved code structure, packaging, documentation, and distribution by implementing an object oriented approach and utilizing modern Python tooling. The goal of this new version is to improve usability and reproducibility, encourage collaboration, and ease maintenance.

[See the changelog](https://github.com/pycoare/coare/blob/main/docs/changelog.md) for all mathematically relevant changes made to the algorithm itself.

**Find more details on the usage and api [in the documentation](https://pycoare.readthedocs.io).**

## Installation

The latest stable version (currently a beta) can be downloaded using pip:
```
pip install pycoare
```
The package can also be added to projects via [uv](https://docs.astral.sh/uv/):
```
uv add pycoare
```
You can install the most up-to-date version using:
```
pip install git+https://github.com/pycoare/coare
```

## Versions

pycoare contains two versions of the COARE algorithm: **COARE v3.5** and **COARE v3.6**.

Version 3.5 was released in 2013, which made adjustments to the wind speed dependence of the Charnock parameter based on a large database of direct covariance stress observations (principally from a buoy) ([Edson et al., 2013](https://doi.org/10.1175/JPO-D-12-0173.1)).
This led to an increase in stress for wind speeds greater than about 18 m/s. The roughness Reynolds number formulation of the scalar roughness length was tuned slightly to give the same values of `Ch` and `Ce` as Version 3.0. The diurnal warm layer model was structured as a separate routine instead of embedded in a driver program. COARE v3.5 was based on buoy data ([Edson et al., 2013](https://doi.org/10.1175/JPO-D-12-0173.1)) and was compared to a large database (a total of 16,000 hours of observations) combining observations from NOAA, WHOI, and U. Miami ([Fairall et al., 2011](http://doi.wiley.com/10.1029/2010JC006884)).

Version 3.6 is slightly restructured and built around improvements in the representation of the effects of waves on fluxes. This includes improved relationships of surface roughness and whitecap fraction on wave parameters ([Fairall et al., 2022](https://doi.org/10.3389/fmars.2022.826606)).

## Contribution

I welcome any contributions - feel free to [raise an issue](https://github.com/pycoare/coare/issues) or submit a [pull request](https://github.com/pycoare/coare/pulls). Take a look at the [contribution guide](https://pycoare.readthedocs.io/en/latest/contributing.html) to get started!

## Credits

This Python implementation of the COARE algorithm was initially translated from MATLAB by
Byron Blomquist, Ludovic Bariteau, with support from the NOAA Physical Sciences Laboratory ([Ludovic et al., 2021](https://zenodo.org/records/5110991)).

The development of the COARE algorithm builds upon decades of research, for which I am extremely appreciative.
The history of the COARE algorithm and its development [can be found here](https://github.com/pyCOARE/coare/tree/main/docs/References) ([Fairall et al., 2022](https://doi.org/10.3389/fmars.2022.826606)).
