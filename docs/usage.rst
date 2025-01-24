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

.. note::

    The COARE v3.5 algorithm can accept NumPy ``ArrayLike`` objects (e.g., ``float``, ``list``, ``numpy.ndarray``, etc.) as input.
    It is not yet designed to work with ``xarray`` objects (support coming soon).
    Please convert any ``xarray.DataArray`` to ``numpy.ndarray`` using the ``.values`` attribute before passing them to the COARE algorithm.

This package is designed around an object oriented approach, since the number of parameters that the COARE algorithm outputs can be quite unwieldy otherwise.
Therefore, the first step in using the COARE algorithm is to create an instance of the ``coare_35`` class.
This instance takes wind speed as the primary argument, but can (and likely should) also take a myriad of additional parameters as keyword arguments.
Here we show an example of how to use the COARE algorithm with only wind speed as input:

    >>> from pycoare import coare_35  # import the coare_35 class
    >>> c = coare_35([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])  # create instance of coare_35 class

Once the ``coare_35`` object is created, the algorithm will automatically be ran.
The output of the algorithm is stored in several subclasses, which can be accessed as attributes from the ``coare_35`` class.
These subclasses should only ever be accessed from within a ``coare_35`` instance (as shown in the examples below).
These classes contain "subattributes" that store the output of the algorithm, neatly divided into categories.

For example, the ``fluxes`` attribute contains the output of the algorithm related to air-sea fluxes.
To access the wind stress (i.e. the air-sea momentum flux), the you can use the ``tau`` subattribute of the ``fluxes`` attribute of the ``coare_35`` object:

    >>> from pycoare import coare_35
    >>> c = coare_35([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])  # same as above example, algorithm is run only once
    >>> c.fluxes.tau  # access the wind stress
    array([0.        , 0.00060093, 0.00165957, 0.00319516, 0.00520412, 0.00768739])

As another example, the ``transfer_coefficients`` attribute contains the output of the algorithm related, unsurprisingly, to transfer coefficients.
To access the neutral wind stress transfer coefficient (i.e., neutral drag coefficient), you can use the ``cdn_rf`` subattribute of the ``transfer_coffiencient`` attribute of the ``coare_35`` object:

    >>> from pycoare import coare_35
    >>> c = coare_35([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])  # same as above example, algorithm is run only once
    >>> c.transfer_coefficients.cdn_rf  # access the wind stress
    array([1.2360058 , 1.12526078, 1.04170881, 0.98860156, 0.95127555, 0.92438611])

The available attribute classes accesseible from a ``coare_35`` instance are:
:class:`fluxes`, :class:`transfer_coefficients`, :class:`stability_functions`, :class:`velocities`, :class:`temperatures`, and :class:`humidities`.
A full list of the available subattributes and a brief description of them can be found in the complete `COARE v3.5 API documentation <c35_api.html>`_.
