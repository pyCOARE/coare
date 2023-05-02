Usage
=====

.. _Usage:

The primary function in the package is the COARE v3.5 algorithm. Usage::

    >>> coare35vn(u) 
    
Include other kwargs as desired


.. autofunction:: pycoare.coare.c35

Vectorized version of COARE 3 code (Fairall et al, 2003) with modification
based on the CLIMODE, MBL and CBLAST experiments (Edson et al., 2013).
The cool skin option is retained but warm layer and surface wave options
have been removed.

This version includes parameterizations of wave height and wave slope using
cp and sigH.  Unless these are provided the wind speed dependent
formulation is used.

Notes:

-  u is the ocean-relative wind speed, i.e., the magnitude of the
   difference between the wind (at zu) and ocean surface current
   vectors.
-  Set jcool=0 if ts is true surface skin temperature,
   otherwise ts is assumed the bulk temperature and jcool=1.
-  The code to compute the heat flux caused by precipitation is
   included if rain data is available (default is no rain).
-  Code updates the cool-skin temperature depression dter and thickness
   tkt during iteration loop for consistency.
-  Number of iterations set to nits = 6.
-  The warm layer is not implemented in this version.
-  Default values are included, but should probably be edited
   to be specific to your region.
   