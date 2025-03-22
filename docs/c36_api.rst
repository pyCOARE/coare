COARE v3.6
==========

.. caution::

   The COARE algorithm is designed to work best when the user has time series of as many input variables as possible.
   This implementation only specifically requires that the ocean surface wind speed ``u`` be provide, but it is highly recommended to use additional variables.
   If any of variables are not available for the full ``u`` time series, a single, representative (e.g., mean) value can be input that will be used in place of the entire time series.
   Please read the documentation carefully to understand each of the input variables to :class:`coare_36`.

.. currentmodule:: pycoare

.. autoclass:: coare_36
   :members:

.. autoclass:: pycoare.coare_36.fluxes

.. autoclass:: pycoare.coare_36.transfer_coefficients

.. autoclass:: pycoare.coare_36.stability_functions

.. autoclass:: pycoare.coare_36.stability_parameters

.. autoclass:: pycoare.coare_36.velocities

.. autoclass:: pycoare.coare_36.temperatures

.. autoclass:: pycoare.coare_36.humidities
