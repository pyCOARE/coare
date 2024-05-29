.. currentmodule:: pycoare

COARE v3.5
==========

.. caution::

   The COARE algorithm is designed to work best when the user has time series of as many input variables as possible. 
   This implementation only specifically requires that the ocean surface wind speed **u** be provided, but 
   it is recommended to provide at least air temperature **t** and relative humidity **rh** as well.
   If any of these variables are not available for the full time series as the wind speed, 
   a single float can be input for any of the other variables and this single value will be used the entire time series, 
   so a representative mean value can be used in place of a full time series. 

.. attention::

   Parameter jcool is used for interpretation of the ocean temperature.
   Set jcool=0 if ts is true surface skin temperature, otherwise use jcool=1 if ts is the bulk temperature (default case).

.. autoclass:: coare_35
   :members:

.. autoclass:: pycoare.coare_35.fluxes

.. autoclass:: pycoare.coare_35.transfer_coefficients

.. autoclass:: pycoare.coare_35.stability_functions

.. autoclass:: pycoare.coare_35.stability_parameters

.. autoclass:: pycoare.coare_35.velocities

.. autoclass:: pycoare.coare_35.temperatures

.. autoclass:: pycoare.coare_35.humidities
