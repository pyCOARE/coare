Installation
============

The latest stable version (currently a beta) can be downloaded using pip::

    >>> pip install pycoare

The package can also be added to projects via `uv <https://docs.astral.sh/uv/>`_::

    >>> uv add pycoare

Dependencies
------------

pycoare requires Python 3.9 or later.

Currently, the only dependency is `numpy <https://numpy.org/>`_ (v2.0 or later).

Future versions may implement support for and require `xarray <https://xarray.pydata.org/en/stable/>`_.

Development Installation
------------------------

Installing the development dependencies is simple using uv::

    >>> uv sync --dev
    >>> .venv/bin/activate

If you are not using uv, you can install the development dependencies using pip::

    >>> pip install -r requirements.txt

See more in the `contribution guide <contributing>`_.
