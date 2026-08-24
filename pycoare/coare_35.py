"""Functions for COARE model bulk flux calculations.

Translated and vectorized from J Edson/C Fairall MATLAB scripts by:

- Byron Blomquist, CU/CIRES, NOAA/ESRL/PSD3
- Ludovic Bariteau, CU/CIRES, NOAA/ESRL/PSD3

Refactored, packaged, and documented by:

- Andrew Scherer, Oregon State University
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr

from .util import (
    NDArrayRealNum,
    _check_size,
    _output,
    _xarray_getters,
    grv,
    psit_26,
    psiu_26,
    psiu_40,
    qair,
    qsea,
    rhcalc,
)


class coare_35:
    """Primary class used for running the COARE v3.5 bulk flux algorithm.

    Usage example using only wind speed as an input (see note above)::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])

    :param u: ocean surface wind speed (m/s) at height zu
    :type u: float | NDArrayRealNum | xr.DataArray
    :param t: bulk air temperature (degC) at height zt
    :type t: float | NDArrayRealNum | xr.DataArray, optional
    :param rh: relative humidity (%) at height zq
    :type rh: float | NDArrayRealNum | xr.DataArray, optional
    :param zu: wind sensor height (m)
    :type zu: float | NDArrayRealNum | xr.DataArray, optional
    :param zt: bulk air temperature sensor height (m)
    :type zt: float | NDArrayRealNum | xr.DataArray, optional
    :param zq: relative humidity sensory height (m)
    :type zq: float | NDArrayRealNum | xr.DataArray, optional
    :param zrf: reference height (m)
    :type zrf: float | NDArrayRealNum | xr.DataArray, optional
    :param us: ocean surface currents (m/s) (defaults to zero, i.e., u is relative wind speed)
    :type us: float | NDArrayRealNum | xr.DataArray, optional
    :param ts: sea water temperature (degC) (also see jcool)
    :type ts: float | NDArrayRealNum | xr.DataArray, optional
    :param p: surface air pressure (mb)
    :type p: float | NDArrayRealNum | xr.DataArray, optional
    :param lat: latitude (deg)
    :type lat: float | NDArrayRealNum | xr.DataArray, optional
    :param zi: planetary boundary layer height (m)
    :type zi: float | NDArrayRealNum | xr.DataArray, optional
    :param rs: downward shortwave radiation (W/m^2)
    :type rs: float | NDArrayRealNum | xr.DataArray, optional
    :param rl: downward longwave radiation (W/m^2)
    :type rl: float | NDArrayRealNum | xr.DataArray, optional
    :param rain: rain rate (mm/hr)
    :type rain: float | NDArrayRealNum | xr.DataArray, optional
    :param cp: phase speed of dominant waves (m/s)
    :type cp: float | NDArrayRealNum | xr.DataArray, optional
    :param sigH: significant wave height (m)
    :type sigH: float | NDArrayRealNum | xr.DataArray, optional
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
        u: float | NDArrayRealNum | xr.DataArray,
        t: float | NDArrayRealNum | xr.DataArray = 10.0,
        rh: float | NDArrayRealNum | xr.DataArray = 75.0,
        zu: float | NDArrayRealNum | xr.DataArray = 10.0,
        zt: float | NDArrayRealNum | xr.DataArray = 10.0,
        zq: float | NDArrayRealNum | xr.DataArray = 10.0,
        zrf: float | NDArrayRealNum | xr.DataArray = 10.0,
        us: float | NDArrayRealNum | xr.DataArray = 0.0,
        ts: float | NDArrayRealNum | xr.DataArray = 10.0,
        p: float | NDArrayRealNum | xr.DataArray = 1015.0,
        lat: float | NDArrayRealNum | xr.DataArray = 45.0,
        zi: float | NDArrayRealNum | xr.DataArray = 600.0,
        rs: float | NDArrayRealNum | xr.DataArray = 150.0,
        rl: float | NDArrayRealNum | xr.DataArray = 370.0,
        rain: float | NDArrayRealNum | xr.DataArray | None = None,
        cp: float | NDArrayRealNum | xr.DataArray | None = None,
        sigH: float | NDArrayRealNum | xr.DataArray | None = None,
        jcool: int = 1,
        nits: int = 10,
    ) -> None:
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

    class _Bulk_Loop_Inputs:
        def __init__(
            self,
            u: float | NDArrayRealNum | xr.DataArray,
            t: float | NDArrayRealNum | xr.DataArray,
            rh: float | NDArrayRealNum | xr.DataArray,
            zu: float | NDArrayRealNum | xr.DataArray,
            zt: float | NDArrayRealNum | xr.DataArray,
            zq: float | NDArrayRealNum | xr.DataArray,
            zrf: float | NDArrayRealNum | xr.DataArray,
            us: float | NDArrayRealNum | xr.DataArray,
            ts: float | NDArrayRealNum | xr.DataArray,
            p: float | NDArrayRealNum | xr.DataArray,
            lat: float | NDArrayRealNum | xr.DataArray,
            zi: float | NDArrayRealNum | xr.DataArray,
            rs: float | NDArrayRealNum | xr.DataArray,
            rl: float | NDArrayRealNum | xr.DataArray,
            rain: float | NDArrayRealNum | xr.DataArray | None,
            cp: float | NDArrayRealNum | xr.DataArray | None,
            sigH: float | NDArrayRealNum | xr.DataArray | None,
            jcool: int,
            nits: int,
        ) -> None:
            # save dimensions and cooordinates if input is xarray DataArray
            self.input_type = type(u)
            if isinstance(u, xr.DataArray):
                self.dims = u.dims
                self.coords = u.coords
                self.u = u.to_numpy()
            else:
                self.dims = None
                self.coords = None
                self.u = u
            self.u = np.asarray(self.u, dtype=np.float64)
            self.shape = self.u.shape
            self.t = _check_size(t, self.shape, "t")
            self.rh = _check_size(rh, self.shape, "rh")
            self.zu = _check_size(zu, self.shape, "zu")
            self.zt = _check_size(zt, self.shape, "zt")
            self.zq = _check_size(zq, self.shape, "zq")
            self.zrf = _check_size(zrf, self.shape, "zrf")
            self.us = _check_size(us, self.shape, "us")
            self.ts = _check_size(ts, self.shape, "ts")
            self.p = _check_size(p, self.shape, "p")
            self.lat = _check_size(lat, self.shape, "Lat")
            self.zi = _check_size(zi, self.shape, "zi")
            self.rs = _check_size(rs, self.shape, "rs")
            self.rl = _check_size(rl, self.shape, "rl")
            self.rain = _check_size(rain, self.shape, "rain")
            # set waveage and seastate flags
            if cp is not None:
                self.waveage_flag = ~np.isnan(cp)
                self.cp = _check_size(cp, self.shape, "cp")
            else:
                self.waveage_flag = False
                self.cp = np.full(self.shape, np.nan, dtype=np.float64)
            if sigH is not None:
                self.seastate_flag = ~np.isnan(sigH) & self.waveage_flag
                self.sigH = _check_size(sigH, self.shape, "sigH")
            else:
                self.seastate_flag = False
                self.sigH = np.full(self.shape, np.nan, dtype=np.float64)
            # all input other than 0 defaults to jcool=1
            self.jcool = 1 if jcool else 0
            self.nits = nits

            # set constants
            self.grav = grv(self.lat)
            self.qs, self.q = self._get_humidities(self.ts, self.t, self.p, self.rh)
            self.lhvap, self.rhoa, self.visa = self._get_air_constants(
                self.ts,
                self.t,
                self.p,
                self.q,
            )
            self.al, self.bigc, self.wetc = self._get_cool_skin(
                self.ts,
                self.grav,
                self.lhvap,
                self.rhoa,
                self.qs,
            )
            self.rns, self.rnl = self._get_radiation_fluxes(
                self.rs,
                self.rl,
                self.ts,
                self.jcool,
            )

        @staticmethod
        def _get_humidities(
            ts: NDArrayRealNum,
            t: NDArrayRealNum,
            p: NDArrayRealNum,
            rh: NDArrayRealNum,
        ) -> tuple[NDArrayRealNum, NDArrayRealNum]:
            return qsea(ts, p) / 1000, qair(t, p, rh) / 1000

        @staticmethod
        def _get_air_constants(
            ts: NDArrayRealNum,
            t: NDArrayRealNum,
            p: NDArrayRealNum,
            q: NDArrayRealNum,
        ) -> tuple[NDArrayRealNum, NDArrayRealNum, NDArrayRealNum]:
            lhvap = (2.501 - 0.00237 * ts) * 1e6
            rhoa = p * 100.0 / (coare_35.RGAS * (t + coare_35.TDK) * (1 + 0.61 * q))
            visa = 1.326e-5 * (1 + 6.542e-3 * t + 8.301e-6 * t**2 - 4.84e-9 * t**3)
            return lhvap, rhoa, visa

        @staticmethod
        def _get_cool_skin(
            ts: NDArrayRealNum,
            grav: NDArrayRealNum,
            lhvap: NDArrayRealNum,
            rhoa: NDArrayRealNum,
            qs: NDArrayRealNum,
        ) -> tuple[NDArrayRealNum, NDArrayRealNum, NDArrayRealNum]:
            al = 2.1e-5 * (ts + 3.2) ** 0.79
            bigc = (
                16.0
                * grav
                * coare_35.CPW
                * (coare_35.RHOW * coare_35.VISW) ** 3
                / (coare_35.TCW**2 * rhoa**2)
            )
            wetc = 0.622 * lhvap * qs / (coare_35.RGAS * (ts + coare_35.TDK) ** 2)
            return al, bigc, wetc

        @staticmethod
        def _get_radiation_fluxes(
            rs: NDArrayRealNum,
            rl: NDArrayRealNum,
            ts: NDArrayRealNum,
            jcool: Literal[0, 1],
        ) -> tuple[NDArrayRealNum, NDArrayRealNum]:
            rns = 0.945 * rs
            rnl = 0.97 * (5.67e-8 * (ts - 0.3 * jcool + coare_35.TDK) ** 4 - rl)
            return (rns, rnl)

    @dataclass
    class _Bulk_Loop_Outputs:
        ut: NDArrayRealNum
        usr: NDArrayRealNum
        tsr: NDArrayRealNum
        qsr: NDArrayRealNum
        du: NDArrayRealNum
        dt: NDArrayRealNum
        dq: NDArrayRealNum
        dter: NDArrayRealNum
        dqer: NDArrayRealNum
        tvsr: NDArrayRealNum
        tssr: NDArrayRealNum
        tkt: NDArrayRealNum
        obukL: NDArrayRealNum
        rnl: NDArrayRealNum
        zet: NDArrayRealNum
        gf: NDArrayRealNum
        zo: NDArrayRealNum
        zot: NDArrayRealNum
        zoq: NDArrayRealNum
        ta: NDArrayRealNum

    def _run(self) -> None:
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
        self.stability_parameters = stability_parameters(
            self._bulk_loop_inputs,
            self._bulk_loop_outputs,
        )
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
        ug = np.array([0.5])
        dter = np.array([0.3])

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
        dter[k50] = dter50
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
        k50 = zetu > 50  # stable with thin M-O length relative to zu

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


@_xarray_getters
class fluxes(_output):
    """Contains the flux outputs computed from the COARE v3.5 algorithm.

    An instance of this class is created whenever a :class:`coare_35` class is created.
    Variables in this class should only be accessed through this instance of the :class:`coare_35` class::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])
        # accessing the Webb correction for latent heat flux
        c.fluxes.hlwebb

    :ivar rnl: net longwave radiation (W/m^2)
    :type rnl: float | NDArrayRealNum | xr.DataArray
    :ivar tau: wind stress (N/m^2)
    :type tau: float | NDArrayRealNum | xr.DataArray
    :ivar hsb: sensible heat flux (W/m^2)
    :type hsb: float | NDArrayRealNum | xr.DataArray
    :ivar hlb: latent heat flux (W/m^2)
    :type hlb: float | NDArrayRealNum | xr.DataArray
    :ivar hbb: buoyancy flux (W/m^2)
    :type hbb: float | NDArrayRealNum | xr.DataArray
    :ivar hsbb: sonic buoyancy flux (W/m^2)
    :type hsbb: float | NDArrayRealNum | xr.DataArray
    :ivar hlwebb: Webb correction for latent heat flux (W/m^2)
    :type hlwebb: float | NDArrayRealNum | xr.DataArray
    :ivar evap: evaporation (mm/hr)
    :type evap: float | NDArrayRealNum | xr.DataArray
    :ivar rf: rain heat flux (W/m^2)
    :type rf: float | NDArrayRealNum | xr.DataArray
    """

    # necessary until support for Python < 3.13 is dropped, when cls.__static_attributes__ can be used instead
    _rnl = None
    _tau = None
    _hsb = None
    _hlb = None
    _hbb = None
    _hsbb = None
    _hlwebb = None
    _evap = None
    _rf = None

    def __init__(self, _bulk_loop_inputs, _bulk_loop_outputs):
        super().__init__(_bulk_loop_inputs, _bulk_loop_outputs)
        # compute fluxes
        self._rnl = _bulk_loop_outputs.rnl  #: upwelling IR radiation (W/m^2)
        self._tau = (
            _bulk_loop_inputs.rhoa * _bulk_loop_outputs.usr**2 / _bulk_loop_outputs.gf
        )
        self._hsb = (
            -_bulk_loop_inputs.rhoa
            * coare_35.CPA
            * _bulk_loop_outputs.usr
            * _bulk_loop_outputs.tsr
        )
        self._hlb = (
            -_bulk_loop_inputs.rhoa
            * _bulk_loop_inputs.lhvap
            * _bulk_loop_outputs.usr
            * _bulk_loop_outputs.qsr
        )
        self._hbb = (
            -_bulk_loop_inputs.rhoa
            * coare_35.CPA
            * _bulk_loop_outputs.usr
            * _bulk_loop_outputs.tvsr
        )
        self._hsbb = (
            -_bulk_loop_inputs.rhoa
            * coare_35.CPA
            * _bulk_loop_outputs.usr
            * _bulk_loop_outputs.tssr
        )
        self._wbar = (
            1.61
            * self._hlb
            / _bulk_loop_inputs.lhvap
            / (1 + 1.61 * _bulk_loop_inputs.q)
            / _bulk_loop_inputs.rhoa
            + self._hsb / _bulk_loop_inputs.rhoa / coare_35.CPA / _bulk_loop_outputs.ta
        )
        self._hlwebb = (
            _bulk_loop_inputs.rhoa
            * self._wbar
            * _bulk_loop_inputs.q
            * _bulk_loop_inputs.lhvap
        )
        self._evap = 1000 * self._hlb / _bulk_loop_inputs.lhvap / 1000 * 3600
        # rain heat flux after Gosnell et al., JGR, 1995
        if _bulk_loop_inputs.rain is None:
            self._rf = np.nan * np.zeros(_bulk_loop_outputs.usr.size)
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
            self._rf = (
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


@_xarray_getters
class velocities(_output):
    """Contains the velocity outputs computed from the COARE v3.5 algorithm.

    An instance of this class is created whenever a :class:`coare_35` class is created.
    Variables in this class should only be accessed through this instance of the :class:`coare_35` class::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])
        # accessing the friction velocity
        c.velocities.usr

    :ivar ut: wind speed at height zt (m/s)
    :type ut: float | NDArrayRealNum | xr.DataArray
    :ivar usr: friction velocity (m/s)
    :type usr: float | NDArrayRealNum | xr.DataArray
    :ivar du: difference between wind speed u and ocean surface current us (m/s)
    :type du: float | NDArrayRealNum | xr.DataArray
    :ivar gf: ratio of du/ut
    :type gf: float | NDArrayRealNum | xr.DataArray
    :ivar u: wind speed at height zu (m/s)
    :type u: float | NDArrayRealNum | xr.DataArray
    :ivar u_rf: wind speed at reference height zrf (m/s)
    :type u_rf: float | NDArrayRealNum | xr.DataArray
    :ivar u_n: neutral wind speed at height zu (m/s)
    :type u_n: float | NDArrayRealNum | xr.DataArray
    :ivar u_n_rf: neutral wind speed at reference height zrf (m/s)
    :type u_n_rf: float | NDArrayRealNum | xr.DataArray
    """

    _ut = None
    _usr = None
    _du = None
    _gf = None
    _u = None
    _u_rf = None
    _u_n = None
    _u_n_rf = None

    def __init__(self, _bulk_loop_inputs, _bulk_loop_outputs, stability_functions):
        super().__init__(_bulk_loop_inputs, _bulk_loop_outputs)
        self._ut = _bulk_loop_outputs.ut
        self._usr = _bulk_loop_outputs.usr
        self._du = _bulk_loop_outputs.du
        self._gf = _bulk_loop_outputs.gf
        self._u = _bulk_loop_outputs.du + _bulk_loop_inputs.us
        self._u_rf = self._u + (
            _bulk_loop_outputs.usr
            / coare_35.VON
            / _bulk_loop_outputs.gf
            * (
                np.log(_bulk_loop_inputs.zrf / _bulk_loop_inputs.zu)
                - stability_functions.psi_u_rf
                + stability_functions.psi_u
            )
        )
        self._u_n = (
            self._u
            + stability_functions.psi_u
            * _bulk_loop_outputs.usr
            / coare_35.VON
            / _bulk_loop_outputs.gf
        )
        self._u_n_rf = (
            self._u_rf
            + stability_functions.psi_u_rf
            * _bulk_loop_outputs.usr
            / coare_35.VON
            / _bulk_loop_outputs.gf
        )


@_xarray_getters
class temperatures(_output):
    """Contains the temperature outputs computed from the COARE v3.5 algorithm.

    An instance of this class is created whenever a :class:`coare_35` class is created.
    Variables in this class should only be accessed through this instance of the :class:`coare_35` class::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])
        # accessing the adiabatic lapse rate
        c.temperatures.lapse

    :ivar lapse: adiabatic lapse rate (K/m)
    :type lapse: float | NDArrayRealNum | xr.DataArray
    :ivar dt: difference between t and ts (K)
    :type dt: float | NDArrayRealNum | xr.DataArray
    :ivar dter: cool skin temperature depression (K)
    :type dter: float | NDArrayRealNum | xr.DataArray
    :ivar t_rf: temperature at reference height zrf (K)
    :type t_rf: float | NDArrayRealNum | xr.DataArray
    :ivar t_n: neutral temperature at height zt (K)
    :type t_n: float | NDArrayRealNum | xr.DataArray
    :ivar t_n_rf: neutral temperature at reference height zrf (K)
    :type t_n_rf: float | NDArrayRealNum | xr.DataArray
    """

    _lapse = None
    _dt = None
    _dter = None
    _t_rf = None
    _t_n = None
    _t_n_rf = None

    def __init__(self, _bulk_loop_inputs, _bulk_loop_outputs, stability_functions):
        super().__init__(_bulk_loop_inputs, _bulk_loop_outputs)
        self._bulk_loop_inputs = _bulk_loop_inputs
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


@_xarray_getters
class humidities(_output):
    """Contains the humidity outputs computed from the COARE v3.5 algorithm.

    An instance of this class is created whenever a :class:`coare_35` class is created.
    Variables in this class should only be accessed through this instance of the :class:`coare_35` class::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])
        # accessing the humidity at height zrf
        c.humidities.q_rf

    :ivar dq: difference between q and qs (g/kg)
    :type dq: float | NDArrayRealNum | xr.DataArray
    :ivar dqer: cool skin humidity depression (g/kg)
    :type dqer: float | NDArrayRealNum | xr.DataArray
    :ivar q_rf: humidity at reference height zrf (g/kg)
    :type q_rf: float | NDArrayRealNum | xr.DataArray
    :ivar q_n: neutral humidity at height zq (g/kg)
    :type q_n: float | NDArrayRealNum | xr.DataArray
    :ivar q_n_rf: neutral humidity at reference height zrf (g/kg)
    :type q_n_rf: float | NDArrayRealNum | xr.DataArray
    :ivar rh_rf: relative humidity at reference height zrf (%)
    :type rh_rf: float | NDArrayRealNum | xr.DataArray
    """

    _dq = None
    _dqer = None
    _q_rf = None
    _q_n = None
    _q_n_rf = None
    _rh_rf = None

    def __init__(
        self,
        _bulk_loop_inputs,
        _bulk_loop_outputs,
        stability_functions,
        temperatures,
    ):
        super().__init__(_bulk_loop_inputs, _bulk_loop_outputs)
        self._dq = _bulk_loop_outputs.dq
        self._dqer = _bulk_loop_outputs.dqer
        self._q_rf = _bulk_loop_inputs.q + _bulk_loop_outputs.qsr / coare_35.VON * (
            np.log(_bulk_loop_inputs.zrf / _bulk_loop_inputs.zq)
            - stability_functions.psi_q_rf
            + stability_functions.psi_t
        )
        self._q_n = _bulk_loop_inputs.q + (
            stability_functions.psi_t
            * _bulk_loop_outputs.qsr
            / coare_35.VON
            / np.sqrt(_bulk_loop_outputs.gf)
        )
        self._q_n_rf = (
            self._q_rf
            + stability_functions.psi_q_rf * _bulk_loop_outputs.qsr / coare_35.VON
        )
        self._rh_rf = rhcalc(temperatures.t_rf, _bulk_loop_inputs.p, self._q_rf)
        # convert to g/kg
        self._dq *= 1000
        self._dqer *= 1000
        self._q_rf *= 1000
        self._q_n *= 1000
        self._q_n_rf *= 1000


@_xarray_getters
class stability_parameters(_output):
    """Contains the stability parameters computed from the COARE v3.5 algorithm.

    An instance of this class is created whenever a :class:`coare_35` class is created.
    Variables in this class should only be accessed through this instance of the :class:`coare_35` class::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])
        # accessing the temperature scaling parameter
        c.stability_parameters.tsr

    :ivar tsr: temperature scaling parameter (K)
    :type tsr: float | NDArrayRealNum | xr.DataArray
    :ivar tvsr: virtual potential temperature scaling parameter (K)
    :type tvsr: float | NDArrayRealNum | xr.DataArray
    :ivar tssr: sonic temperature scaling parameter (K)
    :type tssr: float | NDArrayRealNum | xr.DataArray
    :ivar qsr: humidity scaling parameter (g/kg)
    :type qsr: float | NDArrayRealNum | xr.DataArray
    :ivar tkt: cool skin thickness (m)
    :type tkt: float | NDArrayRealNum | xr.DataArray
    :ivar obukL: Obukhov length scale (m)
    :type obukL: float | NDArrayRealNum | xr.DataArray
    :ivar zet: Monin-Obukhov stability parameter
    :type zet: float | NDArrayRealNum | xr.DataArray
    :ivar zo: roughness length (m)
    :type zo: float | NDArrayRealNum | xr.DataArray
    :ivar zot: thermal roughness length (m)
    :type zot: float | NDArrayRealNum | xr.DataArray
    :ivar zoq: moisture roughness length (m)
    :type zoq: float | NDArrayRealNum | xr.DataArray
    """

    _tsr = None
    _tvsr = None
    _tssr = None
    _qsr = None
    _tkt = None
    _obukL = None
    _zet = None
    _zo = None
    _zot = None
    _zoq = None

    def __init__(self, _bulk_loop_inputs, _bulk_loop_outputs):
        super().__init__(_bulk_loop_inputs, _bulk_loop_outputs)
        self._tsr = _bulk_loop_outputs.tsr
        self._tvsr = _bulk_loop_outputs.tvsr
        self._tssr = _bulk_loop_outputs.tssr
        self._qsr = _bulk_loop_outputs.qsr
        self._tkt = _bulk_loop_outputs.tkt
        self._obukL = _bulk_loop_outputs.obukL
        self._zet = _bulk_loop_outputs.zet
        self._zo = _bulk_loop_outputs.zo
        self._zot = _bulk_loop_outputs.zot
        self._zoq = _bulk_loop_outputs.zoq


@_xarray_getters
class transfer_coefficients(_output):
    """Contains the transfer coefficients computed from the COARE v3.5 algorithm.

    An instance of this class is created whenever a :class:`coare_35` class is created.
    Variables in this class should only be accessed through this instance of the :class:`coare_35` class::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])
        # accessing the wind stress transfer coefficient
        c.transfer_coefficients.cd

    :ivar cd: wind stress transfer (drag) coefficient at height zu
    :type cd: float | NDArrayRealNum | xr.DataArray
    :ivar ch: sensible heat transfer coefficient (Stanton number) at height zu
    :type ch: float | NDArrayRealNum | xr.DataArray
    :ivar ce: latent heat transfer coefficient (Dalton number) at height zu
    :type ce: float | NDArrayRealNum | xr.DataArray
    :ivar cdn_rf: neutral wind stress transfer (drag) coefficient at reference height zrf
    :type cdn_rf: float | NDArrayRealNum | xr.DataArray
    :ivar chn_rf: neutral sensible heat transfer coefficient (Stanton number) at reference height zrf
    :type chn_rf: float | NDArrayRealNum | xr.DataArray
    :ivar cen_rf: neutral latent heat transfer coefficient (Dalton number) at reference height zrf
    :type cen_rf: float | NDArrayRealNum | xr.DataArray
    """

    _cd = None
    _ch = None
    _ce = None
    _cdn_rf = None
    _chn_rf = None
    _cen_rf = None

    def __init__(self, _bulk_loop_inputs, _bulk_loop_outputs, fluxes):
        super().__init__(_bulk_loop_inputs, _bulk_loop_outputs)
        # compute transfer coeffs relative to ut @ meas. ht
        self._cd = (
            fluxes.tau
            / _bulk_loop_inputs.rhoa
            / _bulk_loop_outputs.ut
            / np.maximum(0.1, _bulk_loop_outputs.du)
        )
        self._ch = (
            -_bulk_loop_outputs.usr
            * _bulk_loop_outputs.tsr
            / _bulk_loop_outputs.ut
            / (
                _bulk_loop_outputs.dt
                - _bulk_loop_outputs.dter * _bulk_loop_inputs.jcool
            )
        )
        self._ce = (
            -_bulk_loop_outputs.usr
            * _bulk_loop_outputs.qsr
            / (
                _bulk_loop_outputs.dq
                - _bulk_loop_outputs.dqer * _bulk_loop_inputs.jcool
            )
            / _bulk_loop_outputs.ut
        )
        # compute at ref height zrf neutral coeff relative to ut
        self._cdn_rf = (
            coare_35.VON**2 / np.log(_bulk_loop_inputs.zrf / _bulk_loop_outputs.zo) ** 2
        )
        self._chn_rf = (
            coare_35.VON**2
            * coare_35.FDG
            / np.log(_bulk_loop_inputs.zrf / _bulk_loop_outputs.zo)
            / np.log(_bulk_loop_inputs.zrf / _bulk_loop_outputs.zot)
        )
        self._cen_rf = (
            coare_35.VON**2
            * coare_35.FDG
            / np.log(_bulk_loop_inputs.zrf / _bulk_loop_outputs.zo)
            / np.log(_bulk_loop_inputs.zrf / _bulk_loop_outputs.zoq)
        )


@_xarray_getters
class stability_functions(_output):
    """Contains the stability functions calculated from the COARE v3.5 algorithm.

    An instance of this class is created whenever a :class:`coare_35` class is created.
    Variables in this class should only be accessed through this instance of the :class:`coare_35` class::

        from pycoare import coare_35
        # creating a coare_35 instance
        c = coare_35([1])
        # accessing the velocity structure function
        c.stability_functions.psi_u

    :ivar psi_u: velocity structure function
    :type psi_u: float | NDArrayRealNum | xr.DataArray
    :ivar psi_u_rf: velocity structure function at reference height zrf
    :type psi_u_rf: float | NDArrayRealNum | xr.DataArray
    :ivar psi_t: temperature structure function
    :type psi_t: float | NDArrayRealNum | xr.DataArray
    :ivar psi_t_rf: temperature structure function at reference height zrf
    :type psi_t_rf: float | NDArrayRealNum | xr.DataArray
    :ivar psi_q: moisture structure function
    :type psi_q: float | NDArrayRealNum | xr.DataArray
    :ivar psi_q_rf: moisture structure function at reference height zrf
    :type psi_q_rf: float | NDArrayRealNum | xr.DataArray

    """

    _psi_u = None
    _psi_u_rf = None
    _psi_t = None
    _psi_t_rf = None
    _psi_q = None
    _psi_q_rf = None

    def __init__(self, _bulk_loop_inputs, _bulk_loop_outputs):
        super().__init__(_bulk_loop_inputs, _bulk_loop_outputs)
        # compute the stability functions
        self._psi_u = psiu_26(_bulk_loop_inputs.zu / _bulk_loop_outputs.obukL)
        self._psi_u_rf = psiu_26(_bulk_loop_inputs.zrf / _bulk_loop_outputs.obukL)
        self._psi_t = psit_26(_bulk_loop_inputs.zt / _bulk_loop_outputs.obukL)
        self._psi_t_rf = psit_26(_bulk_loop_inputs.zrf / _bulk_loop_outputs.obukL)
        self._psi_q = psit_26(_bulk_loop_inputs.zq / _bulk_loop_outputs.obukL)
        self._psi_q_rf = psit_26(_bulk_loop_inputs.zrf / _bulk_loop_outputs.obukL)
