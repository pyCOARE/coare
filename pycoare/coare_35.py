"""Functions for COARE model bulk flux calculations.

Translated and vectorized from J Edson/C Fairall MATLAB scripts by:

- Byron Blomquist, CU/CIRES, NOAA/ESRL/PSD3
- Ludovic Bariteau, CU/CIRES, NOAA/ESRL/PSD3

Refactored, packaged, and documented by:

- Andrew Scherer, Oregon State University
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

from .util import _check_size, grv, psit_26, psiu_26, psiu_40, qair, qsea, rhcalc


class coare_35:
    """Primary class used for running the COARE v3.5 bulk flux algorithm.

    Usage example using only wind speed as an input (see note above)::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])

    :param u: ocean surface wind speed (m/s) at height zu
    :type u: ArrayLike
    :param t: bulk air temperature (degC) at height zt
    :type t: ArrayLike, optional
    :param rh: relative humidity (%) at height zq
    :type rh: ArrayLike, optional
    :param zu: wind sensor height (m)
    :type zu: ArrayLike, optional
    :param zt: bulk air temperature sensor height (m)
    :type zt: ArrayLike, optional
    :param zq: relative humidity sensory height (m)
    :type zq: ArrayLike, optional
    :param zrf: reference height (m)
    :type zrf: ArrayLike, optional
    :param us: ocean surface currents (m/s) (defaults to zero, i.e., u is relative wind speed)
    :type us: ArrayLike, optional
    :param ts: sea water temperature (degC) (also see jcool)
    :type ts: ArrayLike, optional
    :param p: surface air pressure (mb)
    :type p: ArrayLike, optional
    :param lat: latitude (deg)
    :type lat: ArrayLike, optional
    :param zi: planetary boundary layer height (m)
    :type zi: ArrayLike, optional
    :param rs: downward shortwave radiation (W/m^2)
    :type rs: ArrayLike, optional
    :param rl: downward longwave radiation (W/m^2)
    :type rl: ArrayLike, optional
    :param rain: rain rate (mm/hr)
    :type rain: ArrayLike, optional
    :param cp: phase speed of dominant waves (m/s)
    :type cp: ArrayLike, optional
    :param sigH: significant wave height (m)
    :type sigH: ArrayLike, optional
    :param jcool: cool skin option, 1 if ts is bulk ocean temperature, 0 if ts is ocean skin temperature
    :type jcool: int, optional
    :param nits: number of iterations of bulk flux loop
    :type nits: int, optional
    :ivar fluxes: instance of the :class:`fluxes` class
    :ivar transfer_coefficients: instance of the :class:`transfer_coefficients` class
    :ivar stability_functions: instance of the :class:`stability_functions` class
    :ivar velocities: instance of the :class:`velocities` class
    :ivar temperatures: instance of the :class:`temperatures` class
    :ivar humidities: instance of the :class:`humidities` class
    :ivar stability_parameters: instance of the :class:`stability_parameters` class
    """

    # set constants
    BETA = 1.2
    VON = 0.4  # von Karman const
    FDG = 1.00  # Turbulent Prandtl number
    TDK = 273.16

    # air constants
    RGAS = 287.1
    CPA = 1004.67

    # cool skin constants
    BE = 0.026
    CPW = 4000.0
    RHOW = 1022.0
    VISW = 1.0e-6
    TCW = 0.6

    # wave-age dependent coefficients
    A = 0.114
    B = 0.622

    # Sea-state/wave-age dependent coefficients
    AD = 0.091
    BD = 2.0

    # Charnock coefficients
    UMAX = 19
    A1 = 0.0017
    A2 = -0.0050

    def __init__(
        self,
        u: ArrayLike,
        t: ArrayLike = (10.0,),
        rh: ArrayLike = (75.0,),
        zu: ArrayLike = (10.0,),
        zt: ArrayLike = (10.0,),
        zq: ArrayLike = (10.0,),
        zrf: ArrayLike = (10.0,),
        us: ArrayLike = (0.0,),
        ts: ArrayLike = (10.0,),
        p: ArrayLike = (1015.0,),
        lat: ArrayLike = (45.0,),
        zi: ArrayLike = (600.0,),
        rs: ArrayLike = (150.0,),
        rl: ArrayLike = (370.0,),
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1,
        nits: int = 10,
    ) -> None:
        # going to fix this after deprecating the static methods, but for now just ignore the type errors
        self.tau = self._instance_tau  # ty:ignore[invalid-assignment]
        self.ustar = self._instance_ustar  # ty:ignore[invalid-assignment]
        self.tstar = self._instance_tstar  # ty:ignore[invalid-assignment]
        self.qstar = self._instance_qstar  # ty:ignore[invalid-assignment]
        self.sensible = self._instance_sensible  # ty:ignore[invalid-assignment]
        self.latent = self._instance_latent  # ty:ignore[invalid-assignment]
        self.buoyancy = self._instance_buoyancy  # ty:ignore[invalid-assignment]
        self.webb = self._instance_webb  # ty:ignore[invalid-assignment]
        self.cd = self._instance_cd  # ty:ignore[invalid-assignment]

        self._bulk_loop_inputs = self._Bulk_Loop_Inputs(
            u,
            t,
            rh,
            zu,
            zt,
            zq,
            zrf,
            us,
            ts,
            p,
            lat,
            zi,
            rs,
            rl,
            rain,
            cp,
            sigH,
            jcool,
            nits,
        )

        self._run()

    @dataclass
    class _Bulk_Loop_Inputs:
        u: ArrayLike
        t: ArrayLike
        rh: ArrayLike
        zu: ArrayLike
        zt: ArrayLike
        zq: ArrayLike
        zrf: ArrayLike
        us: ArrayLike
        ts: ArrayLike
        p: ArrayLike
        lat: ArrayLike
        zi: ArrayLike
        rs: ArrayLike
        rl: ArrayLike
        rain: ArrayLike
        cp: ArrayLike
        sigH: ArrayLike
        jcool: int
        nits: int

        def __post_init__(self):
            self._sanitize()
            # set constants
            self.grav = grv(self.lat)
            self.qs, self.q = self._get_humidities()
            self.lhvap, self.rhoa, self.visa = self._get_air_constants()
            self.al, self.bigc, self.wetc = self._get_cool_skin()
            self.rns, self.rnl = self._get_radiation_fluxes()

        def _sanitize(self):
            self.u = np.asarray(self.u, dtype=np.float64)
            self.shape = self.u.shape
            self.t = _check_size(self.t, self.shape, "t")
            self.rh = _check_size(self.rh, self.shape, "rh")
            self.zu = _check_size(self.zu, self.shape, "zu")
            self.zt = _check_size(self.zq, self.shape, "zq")
            self.zq = _check_size(self.zt, self.shape, "zt")
            self.zrf = _check_size(self.zrf, self.shape, "zrf")
            self.us = _check_size(self.us, self.shape, "us")
            self.ts = _check_size(self.ts, self.shape, "ts")
            self.p = _check_size(self.p, self.shape, "p")
            self.lat = _check_size(self.lat, self.shape, "Lat")
            self.zi = _check_size(self.zi, self.shape, "zi")
            self.rs = _check_size(self.rs, self.shape, "rs")
            self.rl = _check_size(self.rl, self.shape, "rl")
            self.rain = _check_size(self.rain, self.shape, "rain")
            # set waveage and seastate flags
            if self.cp is not None:
                self.waveage_flag = ~np.isnan(self.cp)
                self.cp = _check_size(self.cp, self.shape, "cp")
            else:
                self.waveage_flag = False
                self.cp = np.nan * np.ones(self.shape)
            if self.sigH is not None:
                self.seastate_flag = ~np.isnan(self.sigH) & self.waveage_flag
                self.sigH = _check_size(self.sigH, self.shape, "sigH")
            else:
                self.seastate_flag = False
                self.sigH = np.nan * np.ones(self.shape)
            # check jcool
            if self.jcool != 0:
                self.jcool = 1  # all input other than 0 defaults to jcool=1

        def _get_humidities(self):
            return qsea(self.ts, self.p) / 1000, qair(self.t, self.p, self.rh) / 1000

        def _get_air_constants(self):
            lhvap = (2.501 - 0.00237 * self.ts) * 1e6
            rhoa = (
                self.p
                * 100.0
                / (coare_35.RGAS * (self.t + coare_35.TDK) * (1 + 0.61 * self.q))
            )
            visa = 1.326e-5 * (
                1 + 6.542e-3 * self.t + 8.301e-6 * self.t**2 - 4.84e-9 * self.t**3
            )
            return lhvap, rhoa, visa

        def _get_cool_skin(self):
            al = 2.1e-5 * (self.ts + 3.2) ** 0.79
            bigc = (
                16.0
                * self.grav
                * coare_35.CPW
                * (coare_35.RHOW * coare_35.VISW) ** 3
                / (coare_35.TCW**2 * self.rhoa**2)
            )
            wetc = (
                0.622
                * self.lhvap
                * self.qs
                / (coare_35.RGAS * (self.ts + coare_35.TDK) ** 2)
            )
            return al, bigc, wetc

        def _get_radiation_fluxes(self):
            rns = 0.945 * self.rs
            rnl = 0.97 * (
                5.67e-8 * (self.ts - 0.3 * self.jcool + coare_35.TDK) ** 4 - self.rl
            )
            return (rns, rnl)

    @dataclass
    class _Bulk_Loop_Outputs:
        ut: NDArray[np.float64]
        usr: NDArray[np.float64]
        tsr: NDArray[np.float64]
        qsr: NDArray[np.float64]
        du: NDArray[np.float64]
        dt: NDArray[np.float64]
        dq: NDArray[np.float64]
        dter: NDArray[np.float64]
        dqer: NDArray[np.float64]
        tvsr: NDArray[np.float64]
        tssr: NDArray[np.float64]
        tkt: NDArray[np.float64]
        obukL: NDArray[np.float64]
        rnl: NDArray[np.float64]
        zet: NDArray[np.float64]
        gf: NDArray[np.float64]
        zo: NDArray[np.float64]
        zot: NDArray[np.float64]
        zoq: NDArray[np.float64]
        ta: NDArray[np.float64]

    @staticmethod
    def tau(
        u: ArrayLike,
        t: ArrayLike = (10.0,),
        rh: ArrayLike = (75.0,),
        zu: ArrayLike = (10.0,),
        zt: ArrayLike = (10.0,),
        zq: ArrayLike = (10.0,),
        zrf: ArrayLike = (10.0,),
        us: ArrayLike = (0.0,),
        ts: ArrayLike = (10.0,),
        p: ArrayLike = (1015.0,),
        lat: ArrayLike = (45.0,),
        zi: ArrayLike = (600.0,),
        rs: ArrayLike = (150.0,),
        rl: ArrayLike = (370.0,),
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1,
        nits: int = 10,
    ) -> NDArray[np.float64]:
        """Calculate wind stress (N/m^2) with gustiness.

        .. WARNING::
        The method :func:`coare_35.tau` is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., :attr:`coare_35_instance.fluxes.tau`.

        :param: see inputs to :class:`coare_35`
        :return: wind stress (N/m^2)
        :rtype: NDArray[np.float64]
        """
        warnings.warn(
            "The method coare_35.tau is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.fluxes.tau.",
            DeprecationWarning,
            stacklevel=2,
        )
        coare = coare_35(
            u,
            t,
            rh,
            zu,
            zt,
            zq,
            zrf,
            us,
            ts,
            p,
            lat,
            zi,
            rs,
            rl,
            rain,
            cp,
            sigH,
            jcool,
            nits,
        )
        return coare._return_vars("tau")

    def _instance_tau(self) -> NDArray[np.float64]:
        warnings.warn(
            "The method coare_35.tau is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.fluxes.tau.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._return_vars("tau")

    @staticmethod
    def ustar(
        u: ArrayLike,
        t: ArrayLike = (10.0,),
        rh: ArrayLike = (75.0,),
        zu: ArrayLike = (10.0,),
        zt: ArrayLike = (10.0,),
        zq: ArrayLike = (10.0,),
        zrf: ArrayLike = (10.0,),
        us: ArrayLike = (0.0,),
        ts: ArrayLike = (10.0,),
        p: ArrayLike = (1015.0,),
        lat: ArrayLike = (45.0,),
        zi: ArrayLike = (600.0,),
        rs: ArrayLike = (150.0,),
        rl: ArrayLike = (370.0,),
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1,
        nits: int = 10,
    ) -> NDArray[np.float64]:
        """Calculate friction velocity (m/s) that includes gustiness.

        .. WARNING::
        The method :func:`coare_35.ustar` is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., :attr:`coare_35_instance.velocities.ustar`.

        :param: see inputs to :class:`coare_35`
        :return: friction velocity (m/s)
        :rtype: NDArray[np.float64]
        """
        warnings.warn(
            "The method coare_35.ustar is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.velocities.ustar.",
            DeprecationWarning,
            stacklevel=2,
        )
        coare = coare_35(
            u,
            t,
            rh,
            zu,
            zt,
            zq,
            zrf,
            us,
            ts,
            p,
            lat,
            zi,
            rs,
            rl,
            rain,
            cp,
            sigH,
            jcool,
            nits,
        )
        return coare._return_vars("usr")

    def _instance_ustar(self) -> NDArray[np.float64]:
        warnings.warn(
            "The method coare_35.ustar is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.velocities.ustar.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._return_vars("usr")

    @staticmethod
    def tstar(
        u: ArrayLike,
        t: ArrayLike = (10.0,),
        rh: ArrayLike = (75.0,),
        zu: ArrayLike = (10.0,),
        zt: ArrayLike = (10.0,),
        zq: ArrayLike = (10.0,),
        zrf: ArrayLike = (10.0,),
        us: ArrayLike = (0.0,),
        ts: ArrayLike = (10.0,),
        p: ArrayLike = (1015.0,),
        lat: ArrayLike = (45.0,),
        zi: ArrayLike = (600.0,),
        rs: ArrayLike = (150.0,),
        rl: ArrayLike = (370.0,),
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1,
        nits: int = 10,
    ) -> NDArray[np.float64]:
        """Calculate the temperature scaling parameter.

        .. WARNING::
        The method :func:`coare_35.tstar` is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., :attr:`coare_35_instance.temperatures.tstar`.

        :param: see inputs to :class:`coare_35`
        :return: temperature scaling parameter (K)
        :rtype: NDArray[np.float64]
        """
        warnings.warn(
            "The method coare_35.tstar is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.temperatures.tstar.",
            DeprecationWarning,
            stacklevel=2,
        )
        coare = coare_35(
            u,
            t,
            rh,
            zu,
            zt,
            zq,
            zrf,
            us,
            ts,
            p,
            lat,
            zi,
            rs,
            rl,
            rain,
            cp,
            sigH,
            jcool,
            nits,
        )
        return coare._return_vars("tsr")

    def _instance_tstar(self) -> NDArray[np.float64]:
        warnings.warn(
            "The method coare_35.tstar is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.temperatures.tstar.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._return_vars("tsr")

    @staticmethod
    def qstar(
        u: ArrayLike,
        t: ArrayLike = (10.0,),
        rh: ArrayLike = (75.0,),
        zu: ArrayLike = (10.0,),
        zt: ArrayLike = (10.0,),
        zq: ArrayLike = (10.0,),
        zrf: ArrayLike = (10.0,),
        us: ArrayLike = (0.0,),
        ts: ArrayLike = (10.0,),
        p: ArrayLike = (1015.0,),
        lat: ArrayLike = (45.0,),
        zi: ArrayLike = (600.0,),
        rs: ArrayLike = (150.0,),
        rl: ArrayLike = (370.0,),
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1,
        nits: int = 10,
    ) -> NDArray[np.float64]:
        """Calculate the humiditiy scaling parameter.

        .. WARNING::
        The method :func:`coare_35.qstar` is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., :attr:`coare_35_instance.humidities.qstar`.

        :param: see inputs to :class:`coare_35`
        :return: humidity scaling parameter (g/kg)
        :rtype: NDArray[np.float64]
        """
        warnings.warn(
            "The method coare_35.qstar is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.humidities.qstar.",
            DeprecationWarning,
            stacklevel=2,
        )
        coare = coare_35(
            u,
            t,
            rh,
            zu,
            zt,
            zq,
            zrf,
            us,
            ts,
            p,
            lat,
            zi,
            rs,
            rl,
            rain,
            cp,
            sigH,
            jcool,
            nits,
        )
        return coare._return_vars("qsr")

    def _instance_qstar(self) -> NDArray[np.float64]:
        warnings.warn(
            "The method coare_35.qstar is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.humidities.qstar.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._return_vars("qsr")

    @staticmethod
    def sensible(
        u: ArrayLike,
        t: ArrayLike = (10.0,),
        rh: ArrayLike = (75.0,),
        zu: ArrayLike = (10.0,),
        zt: ArrayLike = (10.0,),
        zq: ArrayLike = (10.0,),
        zrf: ArrayLike = (10.0,),
        us: ArrayLike = (0.0,),
        ts: ArrayLike = (10.0,),
        p: ArrayLike = (1015.0,),
        lat: ArrayLike = (45.0,),
        zi: ArrayLike = (600.0,),
        rs: ArrayLike = (150.0,),
        rl: ArrayLike = (370.0,),
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1,
        nits: int = 10,
    ) -> NDArray[np.float64]:
        """Calculate sensible heat flux (W/m^2) into the ocean.

        .. WARNING::
        The method :func:`coare_35.sensible` is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., :attr:`coare_35_instance.fluxes.sensible`.

        :param: see inputs to :class:`coare_35`
        :return: sensible heat flux (W/m^2)
        :rtype: NDArray[np.float64]
        """
        warnings.warn(
            "The method coare_35.sensible is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.fluxes.sensible.",
            DeprecationWarning,
            stacklevel=2,
        )
        coare = coare_35(
            u,
            t,
            rh,
            zu,
            zt,
            zq,
            zrf,
            us,
            ts,
            p,
            lat,
            zi,
            rs,
            rl,
            rain,
            cp,
            sigH,
            jcool,
            nits,
        )
        return coare._return_vars("hsb")

    def _instance_sensible(self) -> NDArray[np.float64]:
        warnings.warn(
            "The method coare_35.sensible is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.fluxes.sensible.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._return_vars("hsb")

    @staticmethod
    def latent(
        u: ArrayLike,
        t: ArrayLike = (10.0,),
        rh: ArrayLike = (75.0,),
        zu: ArrayLike = (10.0,),
        zt: ArrayLike = (10.0,),
        zq: ArrayLike = (10.0,),
        zrf: ArrayLike = (10.0,),
        us: ArrayLike = (0.0,),
        ts: ArrayLike = (10.0,),
        p: ArrayLike = (1015.0,),
        lat: ArrayLike = (45.0,),
        zi: ArrayLike = (600.0,),
        rs: ArrayLike = (150.0,),
        rl: ArrayLike = (370.0,),
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1,
        nits: int = 10,
    ) -> NDArray[np.float64]:
        """Calculate latent heat flux (W/m^2) into the ocean.

        .. WARNING::
        The method :func:`coare_35.latent` is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., :attr:`coare_35_instance.fluxes.latent`.

        :param: see inputs to :class:`coare_35`
        :return: latent heat flux (W/m^2)
        :rtype: NDArray[np.float64]
        """
        warnings.warn(
            "The method coare_35.latent is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.fluxes.latent.",
            DeprecationWarning,
            stacklevel=2,
        )
        coare = coare_35(
            u,
            t,
            rh,
            zu,
            zt,
            zq,
            zrf,
            us,
            ts,
            p,
            lat,
            zi,
            rs,
            rl,
            rain,
            cp,
            sigH,
            jcool,
            nits,
        )
        return coare._return_vars("hlb")

    def _instance_latent(self) -> NDArray[np.float64]:
        warnings.warn(
            "The method coare_35.latent is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.fluxes.latent.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._return_vars("hlb")

    @staticmethod
    def buoyancy(
        u: ArrayLike,
        t: ArrayLike = (10.0,),
        rh: ArrayLike = (75.0,),
        zu: ArrayLike = (10.0,),
        zt: ArrayLike = (10.0,),
        zq: ArrayLike = (10.0,),
        zrf: ArrayLike = (10.0,),
        us: ArrayLike = (0.0,),
        ts: ArrayLike = (10.0,),
        p: ArrayLike = (1015.0,),
        lat: ArrayLike = (45.0,),
        zi: ArrayLike = (600.0,),
        rs: ArrayLike = (150.0,),
        rl: ArrayLike = (370.0,),
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1,
        nits: int = 10,
    ) -> NDArray[np.float64]:
        """Calculate buoyancy flux (W/m^2) into the ocean.

        .. WARNING::
        The method :func:`coare_35.buoyancy` is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., :attr:`coare_35_instance.fluxes.buoyancy`.

        :param: see inputs to :class:`coare_35`
        :return: buoyancy flux (W/m^2)
        :rtype: NDArray[np.float64]
        """
        warnings.warn(
            "The method coare_35.buoyancy is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.fluxes.buoyancy.",
            DeprecationWarning,
            stacklevel=2,
        )
        coare = coare_35(
            u,
            t,
            rh,
            zu,
            zt,
            zq,
            zrf,
            us,
            ts,
            p,
            lat,
            zi,
            rs,
            rl,
            rain,
            cp,
            sigH,
            jcool,
            nits,
        )
        return coare._return_vars("hbb")

    def _instance_buoyancy(self) -> NDArray[np.float64]:
        warnings.warn(
            "The method coare_35.buoyancy is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.fluxes.buoyancy.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._return_vars("hbb")

    @staticmethod
    def webb(
        u: ArrayLike,
        t: ArrayLike = (10.0,),
        rh: ArrayLike = (75.0,),
        zu: ArrayLike = (10.0,),
        zt: ArrayLike = (10.0,),
        zq: ArrayLike = (10.0,),
        zrf: ArrayLike = (10.0,),
        us: ArrayLike = (0.0,),
        ts: ArrayLike = (10.0,),
        p: ArrayLike = (1015.0,),
        lat: ArrayLike = (45.0,),
        zi: ArrayLike = (600.0,),
        rs: ArrayLike = (150.0,),
        rl: ArrayLike = (370.0,),
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1,
        nits: int = 10,
    ) -> NDArray[np.float64]:
        """Calculate the Webb correction for latent heat flux (W/m^2).

        .. WARNING::
        The method :func:`coare_35.webb` is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., :attr:`coare_35_instance.fluxes.webb`.

        :param: see inputs to :class:`coare_35`
        :return: Webb correction for latent heat flux (W/m^2)
        :rtype: NDArray[np.float64]
        """
        warnings.warn(
            "The method coare_35.webb is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.fluxes.webb.",
            DeprecationWarning,
            stacklevel=2,
        )
        coare = coare_35(
            u,
            t,
            rh,
            zu,
            zt,
            zq,
            zrf,
            us,
            ts,
            p,
            lat,
            zi,
            rs,
            rl,
            rain,
            cp,
            sigH,
            jcool,
            nits,
        )
        return coare._return_vars("hlwebb")

    def _instance_webb(self) -> NDArray[np.float64]:
        warnings.warn(
            "The method coare_35.webb is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.fluxes.webb.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._return_vars("hlwebb")

    @staticmethod
    def cd(
        u: ArrayLike,
        t: ArrayLike = (10.0,),
        rh: ArrayLike = (75.0,),
        zu: ArrayLike = (10.0,),
        zt: ArrayLike = (10.0,),
        zq: ArrayLike = (10.0,),
        zrf: ArrayLike = (10.0,),
        us: ArrayLike = (0.0,),
        ts: ArrayLike = (10.0,),
        p: ArrayLike = (1015.0,),
        lat: ArrayLike = (45.0,),
        zi: ArrayLike = (600.0,),
        rs: ArrayLike = (150.0,),
        rl: ArrayLike = (370.0,),
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1,
        nits: int = 10,
    ) -> NDArray[np.float64]:
        """Calculate the wind stress transfer (drag) coefficient at height zu.

        .. WARNING::
        The method :func:`coare_35.cd` is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., :attr:`coare_35_instance.transfer_coefficients.cd`.

        :param: see inputs to :class:`coare_35`
        :return: drag coefficient
        :rtype: NDArray[np.float64]
        """
        warnings.warn(
            "The method coare_35.cd is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.transfer_coefficients.cd.",
            DeprecationWarning,
            stacklevel=2,
        )
        coare = coare_35(
            u,
            t,
            rh,
            zu,
            zt,
            zq,
            zrf,
            us,
            ts,
            p,
            lat,
            zi,
            rs,
            rl,
            rain,
            cp,
            sigH,
            jcool,
            nits,
        )
        return coare._return_vars("cd")

    def _instance_cd(self) -> NDArray[np.float64]:
        warnings.warn(
            "The method coare_35.cd is deprecated and will be removed in version 1.0. Use instance attributes instead, e.g., coare_35_instance.transfer_coefficients.cd.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._return_vars("cd")

    def _run(self) -> NDArray[np.float64]:
        """Run the COARE bulk flux calculations."""
        self._bulk_loop_outputs = self._bulk_loop()

        self.fluxes = fluxes(self._bulk_loop_inputs, self._bulk_loop_outputs)
        self.transfer_coefficients = transfer_coefficients(
            self._bulk_loop_inputs,
            self._bulk_loop_outputs,
            self.fluxes,
        )
        self.stability_functions = stability_functions(
            self._bulk_loop_inputs,
            self._bulk_loop_outputs,
        )
        self.stability_parameters = stability_parameters(self._bulk_loop_outputs)
        self.velocities = velocities(
            self._bulk_loop_inputs,
            self._bulk_loop_outputs,
            self.stability_functions,
        )
        self.temperatures = temperatures(
            self._bulk_loop_inputs,
            self._bulk_loop_outputs,
            self.stability_functions,
        )
        self.humidities = humidities(
            self._bulk_loop_inputs,
            self._bulk_loop_outputs,
            self.stability_functions,
            self.temperatures,
        )

    def _bulk_loop(self):
        bulk_loop_inputs = self._bulk_loop_inputs
        rnl = bulk_loop_inputs.rnl

        # first guess
        du, dt, dq = self._get_dudtdq()
        ta = bulk_loop_inputs.t + self.TDK
        ug = 0.5
        dter = 0.3

        ut = np.sqrt(du**2 + ug**2)
        u10 = ut * np.log(10 / 1e-4) / np.log(bulk_loop_inputs.zu / 1e-4)
        usr = 0.035 * u10

        zo10, _, zot10 = self._get_roughness(np.nan, usr, setup=True)
        zetu, k50 = self._get_mo_stability_setup(ta, ut, zo10, dt, dq, dter)
        obukL10 = self._get_obukhov_length(zetu)
        usr, tsr, qsr = self._get_star(
            ut,
            dt,
            dq,
            dter,
            zo10,
            zot10,
            np.nan,
            obukL10,
            setup=True,
        )
        tkt = 0.001 * np.ones(bulk_loop_inputs.shape)
        charnC, charnW, charnS = self._get_charn(u10, usr, setup=True)

        for i in range(bulk_loop_inputs.nits):
            zet = (
                self.VON
                * bulk_loop_inputs.grav
                * bulk_loop_inputs.zu
                / ta
                * (tsr + 0.61 * ta * qsr)
                / (usr**2)
            )

            charn = charnC
            charn[bulk_loop_inputs.waveage_flag] = charnW[bulk_loop_inputs.waveage_flag]
            charn[bulk_loop_inputs.seastate_flag] = charnS[
                bulk_loop_inputs.seastate_flag
            ]

            obukL = self._get_obukhov_length(zet)
            zo, zoq, zot = self._get_roughness(charn, usr)
            usr, tsr, qsr = self._get_star(ut, dt, dq, dter, zo, zot, zoq, obukL)
            tssr = tsr + 0.51 * ta * qsr
            tvsr = tsr + 0.61 * ta * qsr

            ug = self._get_ug(ta, usr, tvsr)
            ut = np.sqrt(du**2 + ug**2)
            # probably a better way to do this, but this avoids a divide by zero runtime warning
            gf = np.full(bulk_loop_inputs.shape, np.inf)
            k = du != 0
            gf[k] = ut[k] / du[k]

            tkt, dter, dqer = self._get_cool_skin(usr, tsr, qsr, tkt, rnl)
            rnl = 0.97 * (
                5.67e-8
                * (bulk_loop_inputs.ts - dter * bulk_loop_inputs.jcool + self.TDK) ** 4
                - bulk_loop_inputs.rl
            )

            # save first iteration solution for case of zetu>50
            if i == 0:
                usr50 = usr[k50]
                tsr50 = tsr[k50]
                qsr50 = qsr[k50]
                obukL50 = obukL[k50]
                zet50 = zet[k50]
                dter50 = dter[k50]
                dqer50 = dqer[k50]
                tkt50 = tkt[k50]

            u10N = usr / self.VON / gf * np.log(10 / zo)
            charnC, charnW, charnS = self._get_charn(u10N, usr, bulk_loop_inputs)

        usr[k50] = usr50
        tsr[k50] = tsr50
        qsr[k50] = qsr50
        obukL[k50] = obukL50
        zet[k50] = zet50
        dter[k50] = dter50  # ty:ignore[invalid-assignment]
        dqer[k50] = dqer50
        tkt[k50] = tkt50
        bulk_loop_outputs = self._Bulk_Loop_Outputs(
            ut,
            usr,
            tsr,
            qsr,
            du,
            dt,
            dq,
            dter,
            dqer,
            tvsr,
            tssr,
            tkt,
            obukL,
            rnl,
            zet,
            gf,
            zo,
            zot,
            zoq,
            ta,
        )
        return bulk_loop_outputs

    def _get_dudtdq(self):
        bulk_loop_inputs = self._bulk_loop_inputs
        du = bulk_loop_inputs.u - bulk_loop_inputs.us
        dt = bulk_loop_inputs.ts - bulk_loop_inputs.t - 0.0098 * bulk_loop_inputs.zt
        dq = bulk_loop_inputs.qs - bulk_loop_inputs.q
        return du, dt, dq

    def _get_ug(self, ta, usr, tvsr):
        bulk_loop_inputs = self._bulk_loop_inputs
        Bf = -bulk_loop_inputs.grav / ta * usr * tvsr
        ug = 0.2 * np.ones(bulk_loop_inputs.shape)
        k = Bf > 0
        if bulk_loop_inputs.zrf.size == 1:
            ug[k] = self.BETA * (Bf[k] * bulk_loop_inputs.zi) ** (1 / 3)
        else:
            ug[k] = self.BETA * (Bf[k] * bulk_loop_inputs.zi[k]) ** (1 / 3)
        return ug

    def _get_mo_stability_setup(self, ta, ut, zo, dt, dq, dter):
        bulk_loop_inputs = self._bulk_loop_inputs
        cd10 = (self.VON / np.log(10 / zo)) ** 2
        ch10 = 0.00115
        ct10 = ch10 / np.sqrt(cd10)
        zot10 = 10 / np.exp(self.VON / ct10)
        cd = (self.VON / np.log(bulk_loop_inputs.zu / zo)) ** 2
        ct = self.VON / np.log(bulk_loop_inputs.zt / zot10)
        cc = self.VON * ct / cd
        ribcu = -bulk_loop_inputs.zu / bulk_loop_inputs.zi / 0.004 / self.BETA**3
        ribu = (
            -bulk_loop_inputs.grav
            * bulk_loop_inputs.zu
            / ta
            * ((dt - dter * bulk_loop_inputs.jcool) + 0.61 * ta * dq)
            / ut**2
        )
        zetu = cc * ribu * (1 + 27 / 9 * ribu / cc)
        k50 = zetu > 50  # stable with thin M-O length relative to zu  # noqa: PLR2004

        k = ribu < 0
        if ribcu.size == 1:
            zetu[k] = cc[k] * ribu[k] / (1 + ribu[k] / ribcu)
        else:
            zetu[k] = cc[k] * ribu[k] / (1 + ribu[k] / ribcu[k])
        return zetu, k50

    def _get_charn(self, u, usr, setup=False):  # noqa: FBT002, fine for private function
        bulk_loop_inputs = self._bulk_loop_inputs
        # The following gives the new formulation for the Charnock variable
        charnC = self.A1 * u + self.A2
        k = u > self.UMAX
        charnC[k] = self.A1 * self.UMAX + self.A2
        charnW = self.A * (usr / bulk_loop_inputs.cp) ** self.B
        if setup:
            zoS = (
                bulk_loop_inputs.sigH * self.AD * (usr / bulk_loop_inputs.cp) ** self.BD
            )
        else:
            zoS = (
                bulk_loop_inputs.sigH * self.AD * (usr / bulk_loop_inputs.cp) ** self.BD
                - 0.11 * bulk_loop_inputs.visa / usr
            )
        charnS = zoS * bulk_loop_inputs.grav / usr**2
        return charnC, charnW, charnS

    def _get_roughness(self, charn, usr, setup=False):  # noqa: FBT002, fine for private function
        bulk_loop_inputs = self._bulk_loop_inputs
        if setup:
            zo = (
                0.011 * usr**2 / bulk_loop_inputs.grav
                + 0.11 * bulk_loop_inputs.visa / usr
            )
            cd = (self.VON / np.log(10 / zo)) ** 2
            ch = 0.00115
            ct = ch / np.sqrt(cd)
            zot = 10 / np.exp(self.VON / ct)
            zoq = zot
        else:
            # thermal roughness lengths give Stanton and Dalton numbers that
            # closely approximate COARE 3.0
            zo = (
                charn * usr**2 / bulk_loop_inputs.grav
                + 0.11 * bulk_loop_inputs.visa / usr
            )
            rr = zo * usr / bulk_loop_inputs.visa
            zoq = np.minimum(1.6e-4, 5.8e-5 / rr**0.72)
            zot = zoq
        return zo, zoq, zot

    def _get_obukhov_length(self, zet):
        return self._bulk_loop_inputs.zu / zet

    def _get_star(self, ut, dt, dq, dter, zo, zot, zoq, obukL, setup=False):  # noqa: FBT002, fine for private function
        bulk_loop_inputs = self._bulk_loop_inputs
        if setup:
            # unclear why psiu_40 is used here rather than psiu_26 - only place psiu_40 is used
            usr = (
                ut
                * self.VON
                / (
                    np.log(bulk_loop_inputs.zu / zo)
                    - psiu_40(bulk_loop_inputs.zu / obukL)
                )
            )
            tsr = (
                -(dt - dter * bulk_loop_inputs.jcool)
                * self.VON
                * self.FDG
                / (
                    np.log(bulk_loop_inputs.zt / zot)
                    - psit_26(bulk_loop_inputs.zt / obukL)
                )
            )
            qsr = (
                -(dq - bulk_loop_inputs.wetc * dter * bulk_loop_inputs.jcool)
                * self.VON
                * self.FDG
                / (
                    np.log(bulk_loop_inputs.zq / zot)
                    - psit_26(bulk_loop_inputs.zq / obukL)
                )
            )
        else:
            cdhf = self.VON / (
                np.log(bulk_loop_inputs.zu / zo) - psiu_26(bulk_loop_inputs.zu / obukL)
            )
            cqhf = (
                self.VON
                * self.FDG
                / (
                    np.log(bulk_loop_inputs.zq / zoq)
                    - psit_26(bulk_loop_inputs.zq / obukL)
                )
            )
            cthf = (
                self.VON
                * self.FDG
                / (
                    np.log(bulk_loop_inputs.zt / zot)
                    - psit_26(bulk_loop_inputs.zt / obukL)
                )
            )
            usr = ut * cdhf
            qsr = -(dq - bulk_loop_inputs.wetc * dter * bulk_loop_inputs.jcool) * cqhf
            tsr = -(dt - dter * bulk_loop_inputs.jcool) * cthf
        return usr, tsr, qsr

    def _get_cool_skin(self, usr, tsr, qsr, tkt, rnl):
        bulk_loop_inputs = self._bulk_loop_inputs
        hsb = -bulk_loop_inputs.rhoa * self.CPA * usr * tsr
        hlb = -bulk_loop_inputs.rhoa * bulk_loop_inputs.lhvap * usr * qsr
        qout = rnl + hsb + hlb
        dels = bulk_loop_inputs.rns * (
            0.065 + 11 * tkt - 6.6e-5 / tkt * (1 - np.exp(-tkt / 8.0e-4))
        )
        qcol = qout - dels
        alq = (
            bulk_loop_inputs.al * qcol
            + self.BE * hlb * self.CPW / bulk_loop_inputs.lhvap
        )
        xlamx = 6.0 * np.ones(bulk_loop_inputs.shape)
        tkt = np.minimum(
            0.01,
            xlamx * self.VISW / (np.sqrt(bulk_loop_inputs.rhoa / self.RHOW) * usr),
        )
        k = alq > 0
        xlamx[k] = (
            6 / (1 + (bulk_loop_inputs.bigc[k] * alq[k] / usr[k] ** 4) ** 0.75) ** 0.333
        )
        tkt[k] = (
            xlamx[k]
            * self.VISW
            / (np.sqrt(bulk_loop_inputs.rhoa[k] / self.RHOW) * usr[k])
        )
        dter = qcol * tkt / self.TCW
        dqer = bulk_loop_inputs.wetc * dter
        return tkt, dter, dqer

    def _return_vars(self, out):
        outputs = {}
        outputs.update(dict(vars(self._bulk_loop_inputs).items()))
        outputs.update(dict(vars(self._bulk_loop_outputs).items()))
        outputs.update(dict(vars(self.fluxes).items()))
        outputs.update(dict(vars(self.transfer_coefficients).items()))
        outputs.update(dict(vars(self.velocities).items()))
        outputs.update(dict(vars(self.transfer_coefficients).items()))
        outputs.update(dict(vars(self.velocities).items()))
        outputs.update(dict(vars(self.temperatures).items()))
        outputs.update(dict(vars(self.humidities).items()))
        outputs.update(dict(vars(self.stability_functions).items()))
        outputs.update(dict(vars(self.stability_parameters).items()))
        return outputs[out]


class fluxes:
    """Contains the flux outputs computed from the COARE v3.5 algorithm.

    An instance of this class is created whenever a :class:`coare_35` class is created.
    Variables in this class should only be accessed through this instance of the :class:`coare_35` class::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])
        # accessing the Webb correction for latent heat flux
        c.fluxes.hlwebb

    :ivar rnl: upwelling IR radiataion (W/m^2)
    :type rnl: ArrayLike
    :ivar tau: wind stress (N/m^2)
    :type tau: ArrayLike
    :ivar hsb: sensible heat flux (W/m^2)
    :type hsb: ArrayLike
    :ivar hlb: latent heat flux (W/m^2)
    :type hlb: ArrayLike
    :ivar hbb: buoyancy flux (W/m^2)
    :type hbb: ArrayLike
    :ivar hsbb: sonic buoyancy flux (W/m^2)
    :type hsbb: ArrayLike
    :ivar hlwebb: Webb correction for latent heat flux (W/m^2)
    :type hlwebb: ArrayLike
    :ivar evap: evaporation (mm/hr)
    :type evap: ArrayLike
    :ivar rf: rain heat flux (W/m^2)
    :type rf: ArrayLike
    """

    def __init__(self, _bulk_loop_inputs, _bulk_loop_outputs):
        # compute fluxes
        self.rnl = _bulk_loop_outputs.rnl  #: upwelling IR radiation (W/m^2)
        self.tau = (
            _bulk_loop_inputs.rhoa * _bulk_loop_outputs.usr**2 / _bulk_loop_outputs.gf
        )
        self.hsb = (
            -_bulk_loop_inputs.rhoa
            * coare_35.CPA
            * _bulk_loop_outputs.usr
            * _bulk_loop_outputs.tsr
        )
        self.hlb = (
            -_bulk_loop_inputs.rhoa
            * _bulk_loop_inputs.lhvap
            * _bulk_loop_outputs.usr
            * _bulk_loop_outputs.qsr
        )
        self.hbb = (
            -_bulk_loop_inputs.rhoa
            * coare_35.CPA
            * _bulk_loop_outputs.usr
            * _bulk_loop_outputs.tvsr
        )
        self.hsbb = (
            -_bulk_loop_inputs.rhoa
            * coare_35.CPA
            * _bulk_loop_outputs.usr
            * _bulk_loop_outputs.tssr
        )
        self.wbar = (
            1.61
            * self.hlb
            / _bulk_loop_inputs.lhvap
            / (1 + 1.61 * _bulk_loop_inputs.q)
            / _bulk_loop_inputs.rhoa
            + self.hsb / _bulk_loop_inputs.rhoa / coare_35.CPA / _bulk_loop_outputs.ta
        )
        self.hlwebb = (
            _bulk_loop_inputs.rhoa
            * self.wbar
            * _bulk_loop_inputs.q
            * _bulk_loop_inputs.lhvap
        )
        self.evap = 1000 * self.hlb / _bulk_loop_inputs.lhvap / 1000 * 3600
        # rain heat flux after Gosnell et al., JGR, 1995
        if _bulk_loop_inputs.rain is None:
            self.rf = np.nan * np.zeros(_bulk_loop_outputs.usr.size)
        else:
            # water vapour diffusivity
            dwat = (
                2.11e-5 * ((_bulk_loop_inputs.t + coare_35.TDK) / coare_35.TDK) ** 1.94
            )
            # heat diffusivity
            dtmp = (
                (1 + 3.309e-3 * _bulk_loop_inputs.t - 1.44e-6 * _bulk_loop_inputs.t**2)
                * 0.02411
                / (_bulk_loop_inputs.rhoa * coare_35.CPA)
            )
            # Clausius-Clapeyron
            dqs_dt = (
                _bulk_loop_inputs.q
                * _bulk_loop_inputs.lhvap
                / (coare_35.RGAS * (_bulk_loop_inputs.t + coare_35.TDK) ** 2)
            )
            # wet bulb factor
            alfac = 1 / (
                1
                + 0.622
                * (dqs_dt * _bulk_loop_inputs.lhvap * dwat)
                / (coare_35.CPA * dtmp)
            )
            self.rf = (
                _bulk_loop_inputs.rain
                * alfac
                * coare_35.CPW
                * (
                    (
                        _bulk_loop_inputs.ts
                        - _bulk_loop_inputs.t
                        - _bulk_loop_outputs.dter * _bulk_loop_inputs.jcool
                    )
                    + (
                        _bulk_loop_inputs.qs
                        - _bulk_loop_inputs.q
                        - _bulk_loop_outputs.dqer * _bulk_loop_inputs.jcool
                    )
                    * _bulk_loop_inputs.lhvap
                    / coare_35.CPA
                )
                / 3600
            )


class velocities:
    """Contains the velocity outputs computed from the COARE v3.5 algorithm.

    An instance of this class is created whenever a :class:`coare_35` class is created.
    Variables in this class should only be accessed through this instance of the :class:`coare_35` class::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])
        # accessing the friction velocity
        c.velocities.usr

    :ivar ut: wind speed at height zt (m/s)
    :type ut: ArrayLike
    :ivar usr: friction velocity (m/s)
    :type usr: ArrayLike
    :ivar du: difference between wind speed u and ocean surface current us (m/s)
    :type du: ArrayLike
    :ivar gf: ratio of du/ut
    :type gf: ArrayLike
    :ivar u: wind speed at height zu (m/s)
    :type u: ArrayLike
    :ivar u_rf: wind speed at reference height zrf (m/s)
    :type u_rf: ArrayLike
    :ivar u_n: neutral wind speed at height zu (m/s)
    :type u_n: ArrayLike
    :ivar u_n_rf: neutral wind speed at reference height zrf (m/s)
    :type u_n_rf: ArrayLike
    """

    def __init__(self, _bulk_loop_inputs, _bulk_loop_outputs, stability_functions):
        self.ut = _bulk_loop_outputs.ut
        self.usr = _bulk_loop_outputs.usr
        self.du = _bulk_loop_outputs.du
        self.gf = _bulk_loop_outputs.gf
        self.u = _bulk_loop_outputs.du + _bulk_loop_inputs.us
        self.u_rf = self.u + (
            _bulk_loop_outputs.usr
            / coare_35.VON
            / _bulk_loop_outputs.gf
            * (
                np.log(_bulk_loop_inputs.zrf / _bulk_loop_inputs.zu)
                - stability_functions.psi_u_rf
                + stability_functions.psi_u
            )
        )
        self.u_n = (
            self.u
            + stability_functions.psi_u
            * _bulk_loop_outputs.usr
            / coare_35.VON
            / _bulk_loop_outputs.gf
        )
        self.u_n_rf = (
            self.u_rf
            + stability_functions.psi_u_rf
            * _bulk_loop_outputs.usr
            / coare_35.VON
            / _bulk_loop_outputs.gf
        )


class temperatures:
    """Contains the temperature outputs computed from the COARE v3.5 algorithm.

    An instance of this class is created whenever a :class:`coare_35` class is created.
    Variables in this class should only be accessed through this instance of the :class:`coare_35` class::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])
        # accessing the adiabatic lapse rate
        c.temperatures.lapse

    :ivar lapse: adiabatic lapse rate (K/m)
    :type lapse: ArrayLike
    :ivar dt: difference between t and ts (K)
    :type dt: ArrayLike
    :ivar dter: cool skin temperature depression (K)
    :type dter: ArrayLike
    :ivar t_rf: temperature at reference height zrf (K)
    :type t_rf: ArrayLike
    :ivar t_n: neutral temperature at height zt (K)
    :type t_n: ArrayLike
    :ivar t_n_rf: neutral temperature at reference height zrf (K)
    :type t_n_rf: ArrayLike
    """

    def __init__(self, _bulk_loop_inputs, _bulk_loop_outputs, stability_functions):
        self.lapse = _bulk_loop_inputs.grav / coare_35.CPA
        self.dt = _bulk_loop_outputs.dt
        self.dter = _bulk_loop_outputs.dter
        self.t_rf = (
            _bulk_loop_inputs.t
            + _bulk_loop_outputs.tsr
            / coare_35.VON
            * (
                np.log(_bulk_loop_inputs.zrf / _bulk_loop_inputs.zt)
                - stability_functions.psi_t_rf
                + stability_functions.psi_t
            )
            + self.lapse * (_bulk_loop_inputs.zt - _bulk_loop_inputs.zrf)
        )
        self.t_n = (
            _bulk_loop_inputs.t
            + stability_functions.psi_t * _bulk_loop_outputs.tsr / coare_35.VON
        )
        self.t_n_rf = (
            self.t_rf
            + stability_functions.psi_t_rf * _bulk_loop_outputs.tsr / coare_35.VON
        )


class humidities:
    """Contains the humidity outputs computed from the COARE v3.5 algorithm.

    An instance of this class is created whenever a :class:`coare_35` class is created.
    Variables in this class should only be accessed through this instance of the :class:`coare_35` class::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])
        # accessing the humidity at height zrf
        c.humidities.q_rf

    :ivar dq: difference between q and qs (g/kg)
    :type dq: ArrayLike
    :ivar dqer: cool skin humidity depression (g/kg)
    :type dqer: ArrayLike
    :ivar q_rf: humidity at reference height zrf (g/kg)
    :type q_rf: ArrayLike
    :ivar q_n: neutral humidity at height zq (g/kg)
    :type q_n: ArrayLike
    :ivar q_n_rf: neutral humidity at reference height zrf (g/kg)
    :type q_n_rf: ArrayLike
    :ivar rh_rf: relative humidity at reference height zrf (%)
    :type rh_rf: ArrayLike
    """

    def __init__(
        self,
        _bulk_loop_inputs,
        _bulk_loop_outputs,
        stability_functions,
        temperatures,
    ):
        self.dq = _bulk_loop_outputs.dq
        self.dqer = _bulk_loop_outputs.dqer
        self.q_rf = _bulk_loop_inputs.q + _bulk_loop_outputs.qsr / coare_35.VON * (
            np.log(_bulk_loop_inputs.zrf / _bulk_loop_inputs.zq)
            - stability_functions.psi_q_rf
            + stability_functions.psi_t
        )
        self.q_n = _bulk_loop_inputs.q + (
            stability_functions.psi_t
            * _bulk_loop_outputs.qsr
            / coare_35.VON
            / np.sqrt(_bulk_loop_outputs.gf)
        )
        self.q_n_rf = (
            self.q_rf
            + stability_functions.psi_q_rf * _bulk_loop_outputs.qsr / coare_35.VON
        )
        self.rh_rf = rhcalc(temperatures.t_rf, _bulk_loop_inputs.p, self.q_rf)
        # convert to g/kg
        self.q_rf *= 1000
        self.q_n *= 1000
        self.q_n_rf *= 1000


class stability_parameters:
    """Contains the stability parameters computed from the COARE v3.5 algorithm.

    An instance of this class is created whenever a :class:`coare_35` class is created.
    Variables in this class should only be accessed through this instance of the :class:`coare_35` class::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])
        # accessing the temperature scaling parameter
        c.stability_parameters.tsr

    :ivar tsr: temperature scaling parameter (K)
    :type tsr: ArrayLike
    :ivar tvsr: virtual potential temperature scaling parameter (K)
    :type tvsr: ArrayLike
    :ivar tssr: sonic temperature scaling parameter (K)
    :type tssr: ArrayLike
    :ivar qsr: humidity scaling parameter (g/kg)
    :type qsr: ArrayLike
    :ivar tkt: cool skin thickness (m)
    :type tkt: ArrayLike
    :ivar obukL: Obukhov length scale (m)
    :type obukL: ArrayLike
    :ivar zet: Monin-Obukhov stability parameter
    :type zet: ArrayLike
    :ivar zo: roughness length (m)
    :type zo: ArrayLike
    :ivar zot: thermal roughness length (m)
    :type zot: ArrayLike
    :ivar zoq: moisture roughness length (m)
    :type zoq: ArrayLike
    """

    def __init__(self, _bulk_loop_outputs):
        self.tsr = _bulk_loop_outputs.tsr
        self.tvsr = _bulk_loop_outputs.tvsr
        self.tssr = _bulk_loop_outputs.tssr
        self.qsr = _bulk_loop_outputs.qsr
        self.tkt = _bulk_loop_outputs.tkt
        self.obukL = _bulk_loop_outputs.obukL
        self.zet = _bulk_loop_outputs.zet
        self.zo = _bulk_loop_outputs.zo
        self.zot = _bulk_loop_outputs.zot
        self.zoq = _bulk_loop_outputs.zoq


class transfer_coefficients:
    """Contains the transfer coefficients computed from the COARE v3.5 algorithm.

    An instance of this class is created whenever a :class:`coare_35` class is created.
    Variables in this class should only be accessed through this instance of the :class:`coare_35` class::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])
        # accessing the wind stress transfer coefficient
        c.transfer_coefficients.cd

    :ivar cd: wind stress transfer (drag) coefficient at height zu
    :type cd: ArrayLike
    :ivar ch: sensible heat transfer coefficient (Stanton number) at height zu
    :type ch: ArrayLike
    :ivar ce: latent heat transfer coefficient (Dalton number) at height zu
    :type ce: ArrayLike
    :ivar cdn_rf: neutral wind stress transfer (drag) coefficient at reference height zrf
    :type cdn_rf: ArrayLike
    :ivar chn_rf: neutral sensible heat transfer coefficient (Stanton number) at reference height zrf
    :type chn_rf: ArrayLike
    :ivar cen_rf: neutral latent heat transfer coefficient (Dalton number) at reference height zrf
    :type cen_rf: ArrayLike
    """

    def __init__(self, _bulk_loop_inputs, _bulk_loop_outputs, fluxes):
        # compute transfer coeffs relative to ut @ meas. ht
        self.cd = (
            fluxes.tau
            / _bulk_loop_inputs.rhoa
            / _bulk_loop_outputs.ut
            / np.maximum(0.1, _bulk_loop_outputs.du)
        )
        self.ch = (
            -_bulk_loop_outputs.usr
            * _bulk_loop_outputs.tsr
            / _bulk_loop_outputs.ut
            / (
                _bulk_loop_outputs.dt
                - _bulk_loop_outputs.dter * _bulk_loop_inputs.jcool
            )
        )
        self.ce = (
            -_bulk_loop_outputs.usr
            * _bulk_loop_outputs.qsr
            / (
                _bulk_loop_outputs.dq
                - _bulk_loop_outputs.dqer * _bulk_loop_inputs.jcool
            )
            / _bulk_loop_outputs.ut
        )
        # compute at ref height zrf neutral coeff relative to ut
        self.cdn_rf = (
            coare_35.VON**2 / np.log(_bulk_loop_inputs.zrf / _bulk_loop_outputs.zo) ** 2
        )
        self.chn_rf = (
            coare_35.VON**2
            * coare_35.FDG
            / np.log(_bulk_loop_inputs.zrf / _bulk_loop_outputs.zo)
            / np.log(_bulk_loop_inputs.zrf / _bulk_loop_outputs.zot)
        )
        self.cen_rf = (
            coare_35.VON**2
            * coare_35.FDG
            / np.log(_bulk_loop_inputs.zrf / _bulk_loop_outputs.zo)
            / np.log(_bulk_loop_inputs.zrf / _bulk_loop_outputs.zoq)
        )


class stability_functions:
    """Contains the stability functions calculated from the COARE v3.5 algorithm.

    An instance of this class is created whenever a :class:`coare_35` class is created.
    Variables in this class should only be accessed through this instance of the :class:`coare_35` class::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])
        # accessing the velocity structure function
        c.stability_functions.psi_u

    :ivar psi_u: velocity structure function
    :type psi_u: ArrayLike
    :ivar psi_u_rf: velocity structure function at reference height zrf
    :type psi_u_rf: ArrayLike
    :ivar psi_t: temperature structure function
    :type psi_t: ArrayLike
    :ivar psi_t_rf: temperature structure function at reference height zrf
    :type psi_t_rf: ArrayLike
    :ivar psi_q: moisture structure function
    :type psi_q: ArrayLike
    :ivar psi_q_rf: moisture structure function at reference height zrf
    :type psi_q_rf: ArrayLike

    """

    def __init__(self, _bulk_loop_inputs, _bulk_loop_outputs):
        # compute the stability functions
        self.psi_u = psiu_26(_bulk_loop_inputs.zu / _bulk_loop_outputs.obukL)
        self.psi_u_rf = psiu_26(_bulk_loop_inputs.zrf / _bulk_loop_outputs.obukL)
        self.psi_t = psit_26(_bulk_loop_inputs.zt / _bulk_loop_outputs.obukL)
        self.psi_t_rf = psit_26(_bulk_loop_inputs.zrf / _bulk_loop_outputs.obukL)
        self.psi_q = psit_26(_bulk_loop_inputs.zq / _bulk_loop_outputs.obukL)
        self.psi_q_rf = psit_26(_bulk_loop_inputs.zrf / _bulk_loop_outputs.obukL)
