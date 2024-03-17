Usage
=====

.. _Usage:

The COARE v3.5 algorithm is based on the `COARE 3.0 code <_fairall2003>`_ 
with modifications from the results of the `CLIMODE, MBL and CBLAST experiments <_edson2013>`_.

.. _fairall2003: https://doi.org/10.1175/1520-0442(2003)016<0571:BPOASF>2.0.CO;2
.. _edson2013: https://doi.org/10.1175/JPO-D-12-0173.1

Functions within pycoare can be invoked to access specific variables such as wind stress or friction velocity,
accessed via static functions:

    >>> from pycoare import c35
    >>> c35.tau([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    array([0.        , 0.00060093, 0.00165957, 0.00319516, 0.00520412, 0.00768739])

This is intended to be useful when users need only a single variable. If many variables are desired, or the function needs to be called 
repeatedly (in a loop for example), the object oriented approach is preferred since it runs the COARE algorithm only upon instantiation 
which improves performance. After instantiation, desired variables can be accessed using either object functions (which share names with the static functions):

    >>> from pycoare import c35
    >>> c = c35([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])  # algorith is run only once, right now
    >>> c.tau()  # only prints wind stress, does not run algorithm
    array([0.        , 0.00060093, 0.00165957, 0.00319516, 0.00520412, 0.00768739])

Only a handful of variables are currently accessible by functions, which are all included in this page (if you want a dedicated function added, consider `contributing on GitHub <https://github.com/pyCOARE/coare/issues>`_).
However, all variables used in the bulk flux algorithm are accesible through the object oriented API as instance variables accessible through an instance of the c35 class:

    >>> from pycoare import c35
    >>> c = c35([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    >>> c.fluxes.tau
    array([0.        , 0.00060093, 0.00165957, 0.00319516, 0.00520412, 0.00768739])

See the `COARE v3.5 API documentation <_c35_api>`_ for a complete list of methods and variables.

.. note::

Parameter jcool is used for interpretation of the ocean temperature
Set jcool=0 if ts is true surface skin temperature,
otherwise use jcool=1 if ts is the bulk temperature (default case).

.. attention::

Default values are included, but should probably be edited to be specific to your region.
