COARE v3.5
==========

.. caution::

   The COARE algorithm is designed to work best when the user has time series of as many input variables as possible.
   This implementation only specifically requires that the ocean surface wind speed ``u`` be provided, but it is recommended to provide at least air temperature ``t`` and relative humidity ``rh`` as well.
   If any of variables are not available for the full ``u`` time series, a single float can be input that will be used the entire time series.
   **A representative mean value can (and should) be used when a full time series is not available for any variable.**

.. attention::

   Parameter ``jcool`` is used for interpretation of the ocean temperature ``ts``.
   Set ``jcool = 1`` if ``ts`` is the bulk temperature (default case).
   Set ``jcool = 0`` if ``ts`` is true surface skin temperature.

.. currentmodule:: pycoare

.. autoclass:: coare_35
   :members:

.. autoclass:: pycoare.coare_35.fluxes

.. autoclass:: pycoare.coare_35.transfer_coefficients

.. autoclass:: pycoare.coare_35.stability_functions

.. autoclass:: pycoare.coare_35.stability_parameters

.. autoclass:: pycoare.coare_35.velocities

.. autoclass:: pycoare.coare_35.temperatures

.. autoclass:: pycoare.coare_35.humidities
