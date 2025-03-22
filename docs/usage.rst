Usage
=====

.. tip::
    Both the COARE v3.5 and COARE v3.6 algorithms use the same input variables and provide the same output variables.
    This example shows how to use the COARE v3.5 algorithm, but the same steps can be used for the COARE v3.6 algorithm -
    simply replace ``coare_35`` with ``coare_36`` in any code!

This package is designed around an object oriented approach, since the number of parameters that the COARE algorithm outputs can be quite unwieldy otherwise.
Therefore, the first step in using the COARE algorithm is to create an instance of the ``coare_35`` class:

    >>> from pycoare import coare_35  # import the coare_35 class
    >>> c = coare_35([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])  # create instance of coare_35 class

Note that this (and other examples in this documentation) uses *only* wind speed as input.
This is bad practice, and ideally you should use as many input variables as possible.
See :class:`~pycoare.coare_35` or :class:`~pycoare.coare_36` for a complete list of input variables.

.. note::

    ``coare_35`` and ``coare_36`` can accept NumPy ``ArrayLike`` objects (e.g., ``float``, ``list``, ``numpy.ndarray``, etc.) as input.
    It is not yet designed to work with ``xarray`` objects (support coming soon).
    Please convert any ``xarray.DataArray`` to ``numpy.ndarray`` before passing them to the COARE algorithm.

Once the ``coare_35`` object is created, the algorithm will automatically be ran.
The output of the algorithm is stored in several "output classes", which can be accessed as attributes from the ``coare_35`` class.
Output classes contain "subattributes" that store the output of the algorithm, neatly divided into categories.

For example, the ``fluxes`` output class contains the output of the algorithm related to air-sea fluxes.
To access the wind stress (i.e. the air-sea momentum flux), you can use the ``tau`` subattribute of the ``fluxes`` attribute of the ``coare_35`` object:

    >>> from pycoare import coare_35
    >>> c = coare_35([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])  # create instance of coare_35 class
    >>> c.fluxes.tau  # access the wind stress
    array([0.        , 0.00060093, 0.00165957, 0.00319516, 0.00520412, 0.00768739])

.. important::

    These "output classes" should only ever be accessed from within a ``coare_35`` or ``coare_36`` instance (as shown in the examples).

As another example, the ``transfer_coefficients`` output class contains the output of the algorithm related, unsurprisingly, to transfer coefficients.
To access the neutral wind stress transfer coefficient (i.e., neutral drag coefficient), you can use the ``cdn_rf`` subattribute of the ``transfer_coffiencient`` attribute of the ``coare_35`` object:

    >>> from pycoare import coare_35
    >>> c = coare_35([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])  # create instance of coare_35 class
    >>> c.transfer_coefficients.cdn_rf  # access the drag coefficient
    array([1.2360058 , 1.12526078, 1.04170881, 0.98860156, 0.95127555, 0.92438611])

The available output classes accessible from a :class:`~pycoare.coare_35` instance are:

* :class:`~pycoare.coare_35.fluxes`
* :class:`~pycoare.coare_35.transfer_coefficients`
* :class:`~pycoare.coare_35.stability_functions`
* :class:`~pycoare.coare_35.velocities`
* :class:`~pycoare.coare_35.temperatures`
* :class:`~pycoare.coare_35.humidities`

The same output classes are available from a :class:`~pycoare.coare_36` instance:

* :class:`~pycoare.coare_36.fluxes`
* :class:`~pycoare.coare_36.transfer_coefficients`
* :class:`~pycoare.coare_36.stability_functions`
* :class:`~pycoare.coare_36.velocities`
* :class:`~pycoare.coare_36.temperatures`
* :class:`~pycoare.coare_36.humidities`

The links above will take you to the documentation for each of these output classes.
This contains information on the subattributes that are available in each output class.
