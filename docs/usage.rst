Usage
=====

.. _Usage:

.. |fairall2003| replace:: COARE 3.0 code
.. _fairall2003: https://doi.org/10.1175/1520-0442(2003)016<0571:BPOASF>2.0.CO;2
.. |edson2013| replace:: CLIMODE, MBL and CBLAST experiments
.. _edson2013: https://doi.org/10.1175/JPO-D-12-0173.1

The COARE v3.5 algorithm is based on the |fairall2003|_ with modifications from the results of the |edson2013|_.

.. attention::

    Default values are included for the optional parameters (i.e., other than wind speed), but it is better to use available data for as many variables as possible. 
    If you must use a single measurement for any of the variables, try to use averages that are representative of the region of interest.
    `See the API for more details <c35_api.html>`_.

.. attention::

    The COARE v3.5 algorithm can accept floats or NumPy arrays as input. 
    It is not yet designed to work with xarray objects (support coming soon). 
    Please convert DataArrays to NumPy array using the `.values` attribute before passing them to the COARE algorithm.

Functions within pycoare can be invoked to access specific variables such as wind stress or friction velocity,
accessed via static functions:

    >>> from pycoare import c35
    >>> c35.tau([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])  # algorithm is run and prints wind stress
    array([0.        , 0.00060093, 0.00165957, 0.00319516, 0.00520412, 0.00768739])

This is intended to be useful when users need only a single variable. If many variables are desired, or the function needs to be called 
repeatedly (in a loop for example), the object oriented approach is preferred since it runs the COARE algorithm only upon instantiation 
which improves performance. After instantiation, desired variables can be accessed using either object functions (which share names with the static functions):

    >>> from pycoare import c35
    >>> c = c35([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])  # algorithm is run only once, at this step
    >>> c.tau()  # prints wind stress, does not re-run algorithm
    array([0.        , 0.00060093, 0.00165957, 0.00319516, 0.00520412, 0.00768739])

Only a handful of variables are currently accessible by functions, which are all included in this page (if you want a dedicated function added, consider `contributing on GitHub <https://github.com/pyCOARE/coare/issues>`_).
However, all variables used in the bulk flux algorithm are accesible through the object oriented API as instance variables accessible through an instance of the c35 class:

    >>> from pycoare import c35
    >>> c = c35([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])  # same as above example, algorithm is run only once
    >>> c.fluxes.tau  # prints wind stress, does not re-run algorithm
    array([0.        , 0.00060093, 0.00165957, 0.00319516, 0.00520412, 0.00768739])

See the `COARE v3.5 API documentation <c35_api.html>`_ for a complete list of methods and variables.


