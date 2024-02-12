# TODO: Add docstrings where appropriate
# TODO: Possibly break up bulk loop inputs/bulk loop outputs into smaller classes

"""
Functions for COARE model bulk flux calculations.

Translated and vectorized from J Edson/C Fairall MATLAB scripts by:

- Byron Blomquist, CU/CIRES, NOAA/ESRL/PSD3
- Ludovic Bariteau, CU/CIRES, NOAA/ESRL/PSD3

Refactored, packaged, and documented by:

- Andrew Scherer, Oregon State University
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from .util import _check_size, grv, qsea, qair, psit_26, psiu_26, psiu_40, rhcalc


class c35:

    # set constants
    BETA = 1.2
    VON = 0.4          # von Karman const
    FDG = 1.00         # Turbulent Prandtl number
    TDK = 273.16

    # air constants
    RGAS = 287.1
    CPA = 1004.67

    # cool skin constants
    BE = 0.026
    CPW = 4000.
    RHOW = 1022.
    VISW = 1.e-6
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
        t: ArrayLike = [10.],
        rh: ArrayLike = [75.],
        zu: ArrayLike = [10.],
        zt: ArrayLike = [10.],
        zq: ArrayLike = [10.],
        zrf: ArrayLike = [10.],
        us: ArrayLike = [0.],
        ts: ArrayLike = [10.],
        p: ArrayLike = [1015.],
        lat: ArrayLike = [45.],
        zi: ArrayLike = [600.],
        rs: ArrayLike = [150.],
        rl: ArrayLike = [370.],
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1.,
        nits: int = 10
    ) -> None:

        self.tau = self._instance_tau
        self.ustar = self._instance_ustar
        self.tstar = self._instance_tstar
        self.qstar = self._instance_qstar
        self.sensible = self._instance_sensible
        self.latent = self._instance_latent
        self.buoyancy = self._instance_buoyancy
        self.webb = self._instance_webb
        self.cd = self._instance_cd

        self.bulk_loop_inputs = self._Bulk_Loop_Inputs(
            u, t, rh, zu, zt, zq, zrf, us, ts, p, lat, zi, rs, rl, rain, cp, sigH, jcool, nits
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
            self.N = self.u.size
            self.t = _check_size(self.t, self.N, 't')
            self.rh = _check_size(self.rh, self.N, 'rh')
            self.zu = _check_size(self.zu, self.N, 'zu')
            self.zt = _check_size(self.zq, self.N, 'zq')
            self.zq = _check_size(self.zt, self.N, 'zt')
            self.zrf = _check_size(self.zrf, self.N, 'zrf')
            self.us = _check_size(self.us, self.N, 'us')
            self.ts = _check_size(self.ts, self.N, 'ts')
            self.p = _check_size(self.p, self.N, 'p')
            self.lat = _check_size(self.lat, self.N, 'Lat')
            self.zi = _check_size(self.zi, self.N, 'zi')
            self.rs = _check_size(self.rs, self.N, 'rs')
            self.rl = _check_size(self.rl, self.N, 'rl')
            self.rain = _check_size(self.rain, self.N, 'rain')
            # set waveage and seastate flags
            if self.cp is not None:
                self.waveage_flag = ~np.isnan(self.cp)
                self.cp = _check_size(self.cp, self.N, 'cp')
            else:
                self.waveage_flag = False
                self.cp = np.nan * np.ones(self.N)
            if self.sigH is not None:
                self.seastate_flag = ~np.isnan(self.sigH) & self.waveage_flag
                self.sigH = _check_size(self.sigH, self.N, 'sigH')
            else:
                self.seastate_flag = False
                self.sigH = np.nan * np.ones(self.N)
            # check jcool
            if self.jcool != 0:
                self.jcool = 1   # all input other than 0 defaults to jcool=1

        def _get_humidities(self):
            return qsea(self.ts, self.p)/1000, qair(self.t, self.p, self.rh)/1000

        def _get_air_constants(self):
            lhvap = (2.501 - 0.00237*self.ts) * 1e6
            rhoa = self.p*100. / (c35.RGAS * (self.t + c35.TDK) * (1 + 0.61*self.q))
            visa = 1.326e-5 * (1 + 6.542e-3*self.t + 8.301e-6*self.t**2 - 4.84e-9*self.t**3)
            return lhvap, rhoa, visa

        def _get_cool_skin(self):
            al = 2.1e-5 * (self.ts + 3.2)**0.79
            bigc = 16. * self.grav * c35.CPW * (c35.RHOW * c35.VISW)**3 / (c35.TCW**2 * self.rhoa**2)
            wetc = 0.622 * self.lhvap * self.qs / (c35.RGAS * (self.ts + c35.TDK)**2)
            return al, bigc, wetc

        def _get_radiation_fluxes(self):
            rns = 0.945 * self.rs
            rnl = 0.97 * (5.67e-8 * (self.ts - 0.3*self.jcool + c35.TDK)**4 - self.rl)
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

    class _Fluxes:
        def __init__(self, bulk_loop_inputs, bulk_loop_outputs):
            # compute fluxes
            self.tau = bulk_loop_inputs.rhoa*bulk_loop_outputs.usr**2/bulk_loop_outputs.gf
            self.hsb = -bulk_loop_inputs.rhoa*c35.CPA*bulk_loop_outputs.usr*bulk_loop_outputs.tsr
            self.hlb = (-bulk_loop_inputs.rhoa*bulk_loop_inputs.lhvap
                        * bulk_loop_outputs.usr*bulk_loop_outputs.qsr)
            self.hbb = -bulk_loop_inputs.rhoa*c35.CPA*bulk_loop_outputs.usr*bulk_loop_outputs.tvsr
            self.hsbb = -bulk_loop_inputs.rhoa*c35.CPA*bulk_loop_outputs.usr*bulk_loop_outputs.tssr
            self.wbar = (1.61*self.hlb/bulk_loop_inputs.lhvap
                         / (1+1.61*bulk_loop_inputs.q) / bulk_loop_inputs.rhoa
                         + self.hsb/bulk_loop_inputs.rhoa/c35.CPA/bulk_loop_outputs.ta)
            self.hlwebb = bulk_loop_inputs.rhoa*self.wbar*bulk_loop_inputs.q*bulk_loop_inputs.lhvap
            self.evap = 1000*self.hlb/bulk_loop_inputs.lhvap/1000*3600
            # rain heat flux after Gosnell et al., JGR, 1995
            if bulk_loop_inputs.rain is None:
                self.rf = np.nan*np.zeros(bulk_loop_outputs.usr.size)
            else:
                # water vapour diffusivity
                dwat = 2.11e-5*((bulk_loop_inputs.t + c35.TDK)/c35.TDK)**1.94
                # heat diffusivity
                dtmp = ((1 + 3.309e-3*bulk_loop_inputs.t - 1.44e-6*bulk_loop_inputs.t**2)
                        * 0.02411/(bulk_loop_inputs.rhoa*c35.CPA))
                # Clausius-Clapeyron
                dqs_dt = (bulk_loop_inputs.q*bulk_loop_inputs.lhvap
                          / (c35.RGAS*(bulk_loop_inputs.t + c35.TDK)**2))
                # wet bulb factor
                alfac = 1/(1 + 0.622*(dqs_dt*bulk_loop_inputs.lhvap*dwat)/(c35.CPA*dtmp))
                self.rf = (bulk_loop_inputs.rain*alfac*c35.CPW
                           * ((bulk_loop_inputs.ts - bulk_loop_inputs.t
                               - bulk_loop_outputs.dter*bulk_loop_inputs.jcool)
                              + (bulk_loop_inputs.qs - bulk_loop_inputs.q
                                 - bulk_loop_outputs.dqer*bulk_loop_inputs.jcool)
                              * bulk_loop_inputs.lhvap/c35.CPA)/3600)

    class _TransferCoeffs:
        def __init__(self, bulk_loop_inputs, bulk_loop_outputs, fluxes):
            # compute transfer coeffs relative to ut @ meas. ht
            self.cd = (fluxes.tau/bulk_loop_inputs.rhoa/bulk_loop_outputs.ut
                       / np.maximum(0.1, bulk_loop_outputs.du))
            self.ch = (-bulk_loop_outputs.usr*bulk_loop_outputs.tsr/bulk_loop_outputs.ut
                       / (bulk_loop_outputs.dt - bulk_loop_outputs.dter*bulk_loop_inputs.jcool))
            self.ce = (-bulk_loop_outputs.usr*bulk_loop_outputs.qsr
                       / (bulk_loop_outputs.dq - bulk_loop_outputs.dqer*bulk_loop_inputs.jcool)
                       / bulk_loop_outputs.ut)
            # compute at ref height zrf neutral coeff relative to ut
            self.cdn_rf = (1000*c35.VON**2
                           / np.log(bulk_loop_inputs.zrf/bulk_loop_outputs.zo)**2)
            self.chn_rf = (1000*c35.VON**2 * c35.FDG
                           / np.log(bulk_loop_inputs.zrf/bulk_loop_outputs.zo)
                           / np.log(bulk_loop_inputs.zrf/bulk_loop_outputs.zot))
            self.cen_rf = (1000*c35.VON**2 * c35.FDG
                           / np.log(bulk_loop_inputs.zrf/bulk_loop_outputs.zo)
                           / np.log(bulk_loop_inputs.zrf/bulk_loop_outputs.zoq))

    class _StabilityFunctions:
        def __init__(self, bulk_loop_inputs, bulk_loop_outputs):
            # compute the stability functions
            self.psi_u = psiu_26(bulk_loop_inputs.zu/bulk_loop_outputs.obukL)
            self.psi_u_rf = psiu_26(bulk_loop_inputs.zrf/bulk_loop_outputs.obukL)
            self.psi_t = psit_26(bulk_loop_inputs.zt/bulk_loop_outputs.obukL)
            self.psi_t_rf = psit_26(bulk_loop_inputs.zrf/bulk_loop_outputs.obukL)
            self.psi_q = psit_26(bulk_loop_inputs.zq/bulk_loop_outputs.obukL)
            self.psi_q_rf = psit_26(bulk_loop_inputs.zrf/bulk_loop_outputs.obukL)

    class _AirProperties:
        def __init__(self, bulk_loop_inputs, bulk_loop_outputs, stability_functions):
            # Determine the wind speeds relative to ocean surface
            # Note that usr is the friction velocity that includes
            # gustiness usr = sqrt(cd) S, which is equation (18) in
            # Fairall et al. (1996)
            self.lapse = bulk_loop_inputs.grav/c35.CPA
            self.u = bulk_loop_outputs.du
            self.u_rf = (self.u
                         + (bulk_loop_outputs.usr / c35.VON / bulk_loop_outputs.gf
                            * (np.log(bulk_loop_inputs.zrf / bulk_loop_inputs.zu)
                               - stability_functions.psi_u_rf + stability_functions.psi_u)))
            self.u_n = (self.u + stability_functions.psi_u * bulk_loop_outputs.usr
                        / c35.VON / bulk_loop_outputs.gf)
            self.u_n_rf = (self.u_rf + stability_functions.psi_u_rf * bulk_loop_outputs.usr
                           / c35.VON / bulk_loop_outputs.gf)
            self.t_rf = (bulk_loop_inputs.t
                         + bulk_loop_outputs.tsr/c35.VON
                         * (np.log(bulk_loop_inputs.zrf/bulk_loop_inputs.zt)
                            - stability_functions.psi_t_rf + stability_functions.psi_t)
                         + self.lapse*(bulk_loop_inputs.zt - bulk_loop_inputs.zrf))
            self.t_n = bulk_loop_inputs.t + stability_functions.psi_t*bulk_loop_outputs.tsr/c35.VON
            self.t_n_rf = self.t_rf + stability_functions.psi_t_rf*bulk_loop_outputs.tsr/c35.VON
            self.q_rf = (bulk_loop_inputs.q
                         + bulk_loop_outputs.qsr/c35.VON
                         * np.log(bulk_loop_inputs.zrf/bulk_loop_inputs.zq)
                         - stability_functions.psi_q_rf
                         + stability_functions.psi_t)
            self.q_n = (bulk_loop_inputs.q
                        + (stability_functions.psi_t*bulk_loop_outputs.qsr
                           / c35.VON/np.sqrt(bulk_loop_outputs.gf)))
            self.q_n_rf = self.q_rf + stability_functions.psi_q_rf*bulk_loop_outputs.qsr/c35.VON
            self.rh_rf = rhcalc(self.t_rf, bulk_loop_inputs.p, self.q_rf)
            # convert to g/kg
            self.q_rf *= 1000
            self.q_n *= 1000
            self.q_n_rf *= 1000

    @staticmethod
    def tau(
        u: ArrayLike,
        t: ArrayLike = [10.],
        rh: ArrayLike = [75.],
        zu: ArrayLike = [10.],
        zt: ArrayLike = [10.],
        zq: ArrayLike = [10.],
        zrf: ArrayLike = [10.],
        us: ArrayLike = [0.],
        ts: ArrayLike = [10.],
        p: ArrayLike = [1015.],
        lat: ArrayLike = [45.],
        zi: ArrayLike = [600.],
        rs: ArrayLike = [150.],
        rl: ArrayLike = [370.],
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1.,
        nits: int = 10
    ) -> NDArray[np.float64]:
        """Calculate wind stress (N/m^2) with gustiness.

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
        :return: wind stress (N/m^2)
        :rtype: NDArray[np.float64]
        """
        coare = c35(u, t, rh, zu, zt, zq, zrf, us, ts, p, lat, zi, rs, rl, rain, cp, sigH, jcool, nits)
        return coare._return_vars('tau')

    def _instance_tau(self) -> NDArray[np.float64]:
        return self._return_vars('tau')

    @staticmethod
    def ustar(
        u: ArrayLike,
        t: ArrayLike = [10.],
        rh: ArrayLike = [75.],
        zu: ArrayLike = [10.],
        zt: ArrayLike = [10.],
        zq: ArrayLike = [10.],
        zrf: ArrayLike = [10.],
        us: ArrayLike = [0.],
        ts: ArrayLike = [10.],
        p: ArrayLike = [1015.],
        lat: ArrayLike = [45.],
        zi: ArrayLike = [600.],
        rs: ArrayLike = [150.],
        rl: ArrayLike = [370.],
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1.,
        nits: int = 10
    ) -> NDArray[np.float64]:
        """Calculate friction velocity (m/s) with gustiness.

        :param: see :func:`tau`
        :return: friction velocity (m/s)
        :rtype: NDArray[np.float64]
        """
        coare = c35(u, t, rh, zu, zt, zq, zrf, us, ts, p, lat, zi, rs, rl, rain, cp, sigH, jcool, nits)
        return coare._return_vars('usr')

    def _instance_ustar(self) -> NDArray[np.float64]:
        return self._return_vars('usr')

    @staticmethod
    def tstar(
        u: ArrayLike,
        t: ArrayLike = [10.],
        rh: ArrayLike = [75.],
        zu: ArrayLike = [10.],
        zt: ArrayLike = [10.],
        zq: ArrayLike = [10.],
        zrf: ArrayLike = [10.],
        us: ArrayLike = [0.],
        ts: ArrayLike = [10.],
        p: ArrayLike = [1015.],
        lat: ArrayLike = [45.],
        zi: ArrayLike = [600.],
        rs: ArrayLike = [150.],
        rl: ArrayLike = [370.],
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1.,
        nits: int = 10,
    ) -> NDArray[np.float64]:
        """Calculate buoyancy flux (W/m^2) into the ocean.

        :param: see :func:`tau`
        :return: buoyancy flux (W/m^2)
        :rtype: NDArray[np.float64]
        """
        coare = c35(u, t, rh, zu, zt, zq, zrf, us, ts, p, lat, zi, rs, rl, rain, cp, sigH, jcool, nits)
        return coare._return_vars('tsr')

    def _instance_tstar(self) -> NDArray[np.float64]:
        return self._return_vars('tsr')

    @staticmethod
    def qstar(
        u: ArrayLike,
        t: ArrayLike = [10.],
        rh: ArrayLike = [75.],
        zu: ArrayLike = [10.],
        zt: ArrayLike = [10.],
        zq: ArrayLike = [10.],
        zrf: ArrayLike = [10.],
        us: ArrayLike = [0.],
        ts: ArrayLike = [10.],
        p: ArrayLike = [1015.],
        lat: ArrayLike = [45.],
        zi: ArrayLike = [600.],
        rs: ArrayLike = [150.],
        rl: ArrayLike = [370.],
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1.,
        nits: int = 10,
    ) -> NDArray[np.float64]:
        """Calculate buoyancy flux (W/m^2) into the ocean.

        :param: see :func:`tau`
        :return: buoyancy flux (W/m^2)
        :rtype: NDArray[np.float64]
        """
        coare = c35(u, t, rh, zu, zt, zq, zrf, us, ts, p, lat, zi, rs, rl, rain, cp, sigH, jcool, nits)
        return coare._return_vars('qsr')

    def _instance_qstar(self) -> NDArray[np.float64]:
        return self._return_vars('qsr')

    @staticmethod
    def sensible(
        u: ArrayLike,
        t: ArrayLike = [10.],
        rh: ArrayLike = [75.],
        zu: ArrayLike = [10.],
        zt: ArrayLike = [10.],
        zq: ArrayLike = [10.],
        zrf: ArrayLike = [10.],
        us: ArrayLike = [0.],
        ts: ArrayLike = [10.],
        p: ArrayLike = [1015.],
        lat: ArrayLike = [45.],
        zi: ArrayLike = [600.],
        rs: ArrayLike = [150.],
        rl: ArrayLike = [370.],
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1.,
        nits: int = 10
    ) -> NDArray[np.float64]:
        """Calculate sensible heat flux (W/m^2) into the ocean.

        :param: see :func:`tau`
        :return: sensible heat flux (W/m^2)
        :rtype: NDArray[np.float64]
        """
        coare = c35(u, t, rh, zu, zt, zq, zrf, us, ts, p, lat, zi, rs, rl, rain, cp, sigH, jcool, nits)
        return coare._return_vars('hsb')

    def _instance_sensible(self) -> NDArray[np.float64]:
        return self._return_vars('hsb')

    @staticmethod
    def latent(
        u: ArrayLike,
        t: ArrayLike = [10.],
        rh: ArrayLike = [75.],
        zu: ArrayLike = [10.],
        zt: ArrayLike = [10.],
        zq: ArrayLike = [10.],
        zrf: ArrayLike = [10.],
        us: ArrayLike = [0.],
        ts: ArrayLike = [10.],
        p: ArrayLike = [1015.],
        lat: ArrayLike = [45.],
        zi: ArrayLike = [600.],
        rs: ArrayLike = [150.],
        rl: ArrayLike = [370.],
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1.,
        nits: int = 10
    ) -> NDArray[np.float64]:
        """Calculate latent heat flux (W/m^2) into the ocean.

        :param: see :func:`tau`
        :return: latent heat flux (W/m^2)
        :rtype: NDArray[np.float64]
        """
        coare = c35(u, t, rh, zu, zt, zq, zrf, us, ts, p, lat, zi, rs, rl, rain, cp, sigH, jcool, nits)
        return coare._return_vars('hlb')

    def _instance_latent(self) -> NDArray[np.float64]:
        return self._return_vars('hlb')

    @staticmethod
    def buoyancy(
        u: ArrayLike,
        t: ArrayLike = [10.],
        rh: ArrayLike = [75.],
        zu: ArrayLike = [10.],
        zt: ArrayLike = [10.],
        zq: ArrayLike = [10.],
        zrf: ArrayLike = [10.],
        us: ArrayLike = [0.],
        ts: ArrayLike = [10.],
        p: ArrayLike = [1015.],
        lat: ArrayLike = [45.],
        zi: ArrayLike = [600.],
        rs: ArrayLike = [150.],
        rl: ArrayLike = [370.],
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1.,
        nits: int = 10,
    ) -> NDArray[np.float64]:
        """Calculate buoyancy flux (W/m^2) into the ocean.

        :param: see :func:`tau`
        :return: buoyancy flux (W/m^2)
        :rtype: NDArray[np.float64]
        """
        coare = c35(u, t, rh, zu, zt, zq, zrf, us, ts, p, lat, zi, rs, rl, rain, cp, sigH, jcool, nits)
        return coare._return_vars('hbb')

    def _instance_buoyancy(self) -> NDArray[np.float64]:
        return self._return_vars('hbb')

    @staticmethod
    def webb(
        u: ArrayLike,
        t: ArrayLike = [10.],
        rh: ArrayLike = [75.],
        zu: ArrayLike = [10.],
        zt: ArrayLike = [10.],
        zq: ArrayLike = [10.],
        zrf: ArrayLike = [10.],
        us: ArrayLike = [0.],
        ts: ArrayLike = [10.],
        p: ArrayLike = [1015.],
        lat: ArrayLike = [45.],
        zi: ArrayLike = [600.],
        rs: ArrayLike = [150.],
        rl: ArrayLike = [370.],
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1.,
        nits: int = 10,
    ) -> NDArray[np.float64]:
        """Calculate buoyancy flux (W/m^2) into the ocean.

        :param: see :func:`tau`
        :return: buoyancy flux (W/m^2)
        :rtype: NDArray[np.float64]
        """
        coare = c35(u, t, rh, zu, zt, zq, zrf, us, ts, p, lat, zi, rs, rl, rain, cp, sigH, jcool, nits)
        return coare._return_vars('hlwebb')

    def _instance_webb(self) -> NDArray[np.float64]:
        return self._return_vars('hlwebb')

    @staticmethod
    def cd(
        u: ArrayLike,
        t: ArrayLike = [10.],
        rh: ArrayLike = [75.],
        zu: ArrayLike = [10.],
        zt: ArrayLike = [10.],
        zq: ArrayLike = [10.],
        zrf: ArrayLike = [10.],
        us: ArrayLike = [0.],
        ts: ArrayLike = [10.],
        p: ArrayLike = [1015.],
        lat: ArrayLike = [45.],
        zi: ArrayLike = [600.],
        rs: ArrayLike = [150.],
        rl: ArrayLike = [370.],
        rain: ArrayLike = None,
        cp: ArrayLike = None,
        sigH: ArrayLike = None,
        jcool: int = 1.,
        nits: int = 10,
    ) -> NDArray[np.float64]:
        """Calculate buoyancy flux (W/m^2) into the ocean.

        :param: see :func:`tau`
        :return: buoyancy flux (W/m^2)
        :rtype: NDArray[np.float64]
        """
        coare = c35(u, t, rh, zu, zt, zq, zrf, us, ts, p, lat, zi, rs, rl, rain, cp, sigH, jcool, nits)
        return coare._return_vars('cd')

    def _instance_cd(self) -> NDArray[np.float64]:
        return self._return_vars('cd')

    def _run(self) -> NDArray[np.float64]:
        """Run the COARE bulk flux calculations.
        """
        self.bulk_loop_outputs = self._bulk_loop()

        self.fluxes = self._Fluxes(
            self.bulk_loop_inputs,
            self.bulk_loop_outputs
        )
        self.transfer_coeffs = self._TransferCoeffs(
            self.bulk_loop_inputs,
            self.bulk_loop_outputs,
            self.fluxes
        )
        self.stability_functions = self._StabilityFunctions(
            self.bulk_loop_inputs,
            self.bulk_loop_outputs
        )
        self.air_properties = self._AirProperties(
            self.bulk_loop_inputs,
            self.bulk_loop_outputs,
            self.stability_functions
        )

    def _bulk_loop(self):

        bulk_loop_inputs = self.bulk_loop_inputs
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
        usr, tsr, qsr = self._get_star(ut, dt, dq, dter, zo10, zot10, np.nan, obukL10, setup=True)
        tkt = 0.001 * np.ones(bulk_loop_inputs.N)
        charnC, charnW, charnS = self._get_charn(u10, usr, setup=True)

        for i in range(bulk_loop_inputs.nits):
            zet = (self.VON * bulk_loop_inputs.grav * bulk_loop_inputs.zu
                   / ta * (tsr + 0.61 * ta * qsr)
                   / (usr**2))

            charn = charnC
            charn[bulk_loop_inputs.waveage_flag] = charnW[bulk_loop_inputs.waveage_flag]
            charn[bulk_loop_inputs.seastate_flag] = charnS[bulk_loop_inputs.seastate_flag]

            obukL = self._get_obukhov_length(zet)
            zo, zoq, zot = self._get_roughness(charn, usr)
            usr, tsr, qsr = self._get_star(ut, dt, dq, dter, zo, zot, zoq, obukL)
            tssr = tsr + 0.51 * ta * qsr
            tvsr = tsr + 0.61 * ta * qsr

            ug = self._get_ug(ta, usr, tvsr)
            ut = np.sqrt(du**2 + ug**2)
            # probably a better way to do this, but this avoids a divide by zero runtime warning
            gf = np.full(bulk_loop_inputs.N, np.inf)
            k = np.flatnonzero(du != 0)
            gf[k] = ut[k] / du[k]

            tkt, dter, dqer = self._get_cool_skin(usr, tsr, qsr, tkt, rnl)
            rnl = 0.97*(5.67e-8 * (bulk_loop_inputs.ts - dter * bulk_loop_inputs.jcool + self.TDK)**4
                        - bulk_loop_inputs.rl)

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
        dter[k50] = dter50
        dqer[k50] = dqer50
        tkt[k50] = tkt50
        bulk_loop_outputs = self._Bulk_Loop_Outputs(
            ut, usr, tsr, qsr, du, dt, dq, dter, dqer, tvsr, tssr, tkt, obukL, rnl, zet, gf, zo, zot, zoq, ta
        )
        return bulk_loop_outputs

    def _get_dudtdq(self):
        bulk_loop_inputs = self.bulk_loop_inputs
        du = bulk_loop_inputs.u - bulk_loop_inputs.us
        dt = bulk_loop_inputs.ts - bulk_loop_inputs.t - 0.0098 * bulk_loop_inputs.zt
        dq = bulk_loop_inputs.qs - bulk_loop_inputs.q
        return du, dt, dq

    def _get_ug(self, ta, usr, tvsr):
        bulk_loop_inputs = self.bulk_loop_inputs
        Bf = -bulk_loop_inputs.grav / ta * usr * tvsr
        ug = 0.2 * np.ones(bulk_loop_inputs.N)
        k = np.flatnonzero(Bf > 0)
        if bulk_loop_inputs.zrf.size == 1:
            ug[k] = self.BETA * (Bf[k] * bulk_loop_inputs.zi)**(1/3)
        else:
            ug[k] = self.BETA * (Bf[k] * bulk_loop_inputs.zi[k])**(1/3)
        return ug

    def _get_mo_stability_setup(self, ta, ut, zo, dt, dq, dter):
        bulk_loop_inputs = self.bulk_loop_inputs
        cd10 = (self.VON / np.log(10/zo))**2
        ch10 = 0.00115
        ct10 = ch10 / np.sqrt(cd10)
        zot10 = 10 / np.exp(self.VON/ct10)
        cd = (self.VON / np.log(bulk_loop_inputs.zu / zo))**2
        ct = self.VON / np.log(bulk_loop_inputs.zt / zot10)
        cc = self.VON * ct/cd
        ribcu = -bulk_loop_inputs.zu / bulk_loop_inputs.zi / 0.004 / self.BETA**3
        ribu = (-bulk_loop_inputs.grav * bulk_loop_inputs.zu/ta
                * ((dt - dter*bulk_loop_inputs.jcool) + 0.61*ta*dq) / ut**2)
        zetu = cc * ribu * (1 + 27/9 * ribu/cc)
        k50 = np.flatnonzero(zetu > 50)   # stable with thin M-O length relative to zu

        k = np.flatnonzero(ribu < 0)
        if ribcu.size == 1:
            zetu[k] = cc[k] * ribu[k] / (1 + ribu[k] / ribcu)
        else:
            zetu[k] = cc[k] * ribu[k] / (1 + ribu[k] / ribcu[k])
        return zetu, k50

    def _get_charn(self, u, usr, setup=False):
        bulk_loop_inputs = self.bulk_loop_inputs
        # The following gives the new formulation for the Charnock variable
        charnC = self.A1 * u + self.A2
        k = np.flatnonzero(u > self.UMAX)
        charnC[k] = self.A1 * self.UMAX + self.A2
        charnW = self.A * (usr/bulk_loop_inputs.cp)**self.B
        if setup:
            zoS = bulk_loop_inputs.sigH * self.AD * (usr/bulk_loop_inputs.cp)**self.BD
        else:
            zoS = (bulk_loop_inputs.sigH * self.AD * (usr/bulk_loop_inputs.cp)**self.BD
                   - 0.11 * bulk_loop_inputs.visa/usr)
        charnS = zoS * bulk_loop_inputs.grav / usr**2
        return charnC, charnW, charnS

    def _get_roughness(self, charn, usr, setup=False):
        bulk_loop_inputs = self.bulk_loop_inputs
        if setup:
            zo = 0.011 * usr**2 / bulk_loop_inputs.grav + 0.11 * bulk_loop_inputs.visa / usr
            cd = (self.VON / np.log(10/zo))**2
            ch = 0.00115
            ct = ch / np.sqrt(cd)
            zot = 10 / np.exp(self.VON/ct)
            zoq = zot
        else:
            # thermal roughness lengths give Stanton and Dalton numbers that
            # closely approximate COARE 3.0
            zo = charn*usr**2 / bulk_loop_inputs.grav + 0.11*bulk_loop_inputs.visa / usr
            rr = zo*usr / bulk_loop_inputs.visa
            zoq = np.minimum(1.6e-4, 5.8e-5 / rr**0.72)
            zot = zoq
        return zo, zoq, zot

    def _get_obukhov_length(self, zet):
        return self.bulk_loop_inputs.zu / zet

    def _get_star(self, ut, dt, dq, dter, zo, zot, zoq, obukL, setup=False):
        bulk_loop_inputs = self.bulk_loop_inputs
        if setup:
            # unclear why psiu_40 is used here rather than psiu_26 - only place psiu_40 is used
            usr = ut * self.VON / (np.log(bulk_loop_inputs.zu / zo) - psiu_40(bulk_loop_inputs.zu / obukL))
            tsr = (-(dt - dter*bulk_loop_inputs.jcool) * self.VON * self.FDG
                   / (np.log(bulk_loop_inputs.zt / zot) - psit_26(bulk_loop_inputs.zt / obukL)))
            qsr = (-(dq - bulk_loop_inputs.wetc * dter * bulk_loop_inputs.jcool) * self.VON * self.FDG
                   / (np.log(bulk_loop_inputs.zq / zot) - psit_26(bulk_loop_inputs.zq / obukL)))
        else:
            cdhf = (self.VON / (np.log(bulk_loop_inputs.zu / zo)
                                - psiu_26(bulk_loop_inputs.zu / obukL)))
            cqhf = (self.VON*self.FDG / (np.log(bulk_loop_inputs.zq / zoq)
                                         - psit_26(bulk_loop_inputs.zq / obukL)))
            cthf = (self.VON*self.FDG / (np.log(bulk_loop_inputs.zt / zot)
                                         - psit_26(bulk_loop_inputs.zt / obukL)))
            usr = ut*cdhf
            qsr = -(dq - bulk_loop_inputs.wetc * dter * bulk_loop_inputs.jcool) * cqhf
            tsr = -(dt - dter * bulk_loop_inputs.jcool)*cthf
        return usr, tsr, qsr

    def _get_cool_skin(self, usr, tsr, qsr, tkt, rnl):
        bulk_loop_inputs = self.bulk_loop_inputs
        hsb = -bulk_loop_inputs.rhoa*self.CPA*usr*tsr
        hlb = -bulk_loop_inputs.rhoa*bulk_loop_inputs.lhvap*usr*qsr
        qout = rnl + hsb + hlb
        dels = bulk_loop_inputs.rns * (0.065 + 11 * tkt - 6.6e-5 / tkt * (1 - np.exp(-tkt / 8.0e-4)))
        qcol = qout - dels
        alq = bulk_loop_inputs.al * qcol + self.BE * hlb * self.CPW / bulk_loop_inputs.lhvap
        xlamx = 6.0 * np.ones(bulk_loop_inputs.N)
        tkt = np.minimum(0.01, xlamx * self.VISW / (np.sqrt(bulk_loop_inputs.rhoa / self.RHOW) * usr))
        k = np.flatnonzero(alq > 0)
        xlamx[k] = 6 / (1 + (bulk_loop_inputs.bigc[k] * alq[k] / usr[k]**4)**0.75)**0.333
        tkt[k] = xlamx[k] * self.VISW / (np.sqrt(bulk_loop_inputs.rhoa[k] / self.RHOW) * usr[k])
        dter = qcol * tkt / self.TCW
        dqer = bulk_loop_inputs.wetc * dter
        return tkt, dter, dqer

    def _return_vars(self, out):
        outputs = {}
        outputs.update({key: value for key, value in vars(self.bulk_loop_inputs).items()})
        outputs.update({key: value for key, value in vars(self.bulk_loop_outputs).items()})
        outputs.update({key: value for key, value in vars(self.fluxes).items()})
        outputs.update({key: value for key, value in vars(self.transfer_coeffs).items()})
        outputs.update({key: value for key, value in vars(self.air_properties).items()})
        return outputs[out]
