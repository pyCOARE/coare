"""Utility functions for pyCOARE.

Includes functions for calculating normal gravity, relative humidity, saturation vapor pressure, saturation specific humidity at the sea surface, specific humidity from relative humidity, and stability functions for the COARE algorithm.
"""

from __future__ import annotations

import pkgutil
import warnings
from typing import Any, Callable

import numpy as np
import xarray as xr
import yaml
from numpy.typing import NDArray

NDArrayRealNum = NDArray[np.integer] | NDArray[np.floating]


def grv(lat: float | NDArrayRealNum) -> NDArrayRealNum:
    """Compute normal gravity at latitude lat (degrees) using the WGS84 ellipsoid.

    :param lat: latitude (degrees)
    :type lat: float | NDArrayRealNum
    :return: normal gravity (m/s^2)
    :rtype: NDArrayRealNum
    """
    lat = np.deg2rad(np.asarray(lat, dtype=float))
    e = 8.1819190842622e-2  # first eccentricity of Earth
    a = 6378137  # semi-major Earth axis (m)
    b = 6356752.314  # semi-minor Earth axis (m)
    gamma_p = 9.8321849379  # normal gravity at the pole (m/s^2)
    gamma_e = 9.7803253359  # normal gravity at the equator (m/s^2)
    k = b * gamma_p / (a * gamma_e) - 1
    gamma = gamma_e * (1 + k * np.sin(lat) ** 2) / np.sqrt(1 - e**2 * np.sin(lat) ** 2)
    return gamma


def rhcalc(
    t: float | NDArrayRealNum,
    p: float | NDArrayRealNum,
    q: float | NDArrayRealNum,
) -> NDArrayRealNum:
    """Compute relative humidity from temperature, pressure, and specific humidity.

    :param t: temperature (degC)
    :type t: float | NDArrayRealNum
    :param p: pressure (mb)
    :type p: float | NDArrayRealNum
    :param q: specific humidity (g/kg)
    :type q: float | NDArrayRealNum
    :return: relative humidity (%)
    :rtype: NDArrayRealNum
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    es = qsat(t, p)
    em = p * q / (0.622 + 0.378 * q)
    rh = 100.0 * em / es
    return rh


def qsat(t: float | NDArrayRealNum, p: float | NDArrayRealNum) -> NDArrayRealNum:
    """Compute saturation vapor pressure from temperature and pressure.

    :param t: temperature (degC)
    :type t: float | NDArrayRealNum
    :param p: pressure (mb)
    :type p: float | NDArrayRealNum
    :return: saturation vapor pressure (g/kg)
    :rtype: NDArrayRealNum
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    es = 6.1121 * np.exp(17.502 * t / (240.97 + t))
    es *= 1.0007 + p * 3.46e-6
    return es


def qsea(
    t: float | NDArrayRealNum,
    p: float | NDArrayRealNum,
    s: float | NDArrayRealNum = 35.0,
) -> NDArrayRealNum:
    """Compute saturation specific humidity at sea surface from temperature and pressure.

    :param t: temperature (degC)
    :type t: float | NDArrayRealNum
    :param p: pressure (mb)
    :type p: float | NDArrayRealNum
    :return: saturation specific humidity (g/kg)
    :rtype: NDArrayRealNum
    """
    ex = qsat(t, p)  # returns ex as ndarray float
    es = (1 - 0.02 * s / 35) * ex
    qs = 622 * es / (p - 0.378 * es)
    return qs


def qair(
    t: float | NDArrayRealNum,
    p: float | NDArrayRealNum,
    rh: float | NDArrayRealNum,
) -> NDArrayRealNum:
    """Compute specific humidity given temperature, pressure, and relative humidity.

    :param t: temperature (degC)
    :type t: float | NDArrayRealNum
    :param p: pressure (mb)
    :type p: float | NDArrayRealNum
    :param rh: relative humidity (%)
    :type rh: float | NDArrayRealNum
    :return: specific humidity (g/kg), partial pressure (mb)
    :rtype: NDArrayRealNum
    """
    rh = np.asarray(rh, dtype=float)
    rh /= 100.0
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=float)
    em = rh * qsat(t, p=p)
    qa = 621.97 * em / (p - 0.378 * em)
    return qa


def psit_26(z_L: float | NDArrayRealNum) -> NDArrayRealNum:
    """Compute the temperature structure function given z/L.

    :param z_L: stability parameter
    :type z_L: float | NDArrayRealNum
    :return: temperature structure function
    :rtype: NDArrayRealNum
    """
    zet = np.asarray(z_L, dtype=np.float64)
    # compute psi_t for stable conditions by Beljaars & Holtslag 1991
    a = 1
    b = 0.6667
    c = 5
    d = 0.35
    dzet = d * zet
    dzet[dzet > 50] = 50.0
    psi = np.nan * np.empty(zet.shape, dtype=np.float64)
    k = zet >= 0
    psi[k] = -(
        (1 + 2 / 3 * a * zet[k]) ** (3 / 2)
        + b * (zet[k] - c / d) * np.exp(-dzet[k])
        + b * c / d
        - 1
    )
    # compute convective psi_t for unstable conditions by Grachev et. al., 2000
    k = zet < 0
    x = (1 - 15 * zet[k]) ** (1 / 2)
    psik = 2 * np.log((1 + x) / 2.0)  # kansas psi
    x = (1 - 34.15 * zet[k]) ** (1 / 3)
    psic = (
        3 / 2 * np.log((x**2 + x + 1) / 3)  # free convective psi
        - np.sqrt(3) * np.arctan((2 * x + 1) / np.sqrt(3))
        + np.pi / np.sqrt(3)
    )
    # combine free convective and kansas psi
    f = zet[k] ** 2 / (1.0 + zet[k] ** 2.0)
    psi[k] = (1 - f) * psik + f * psic
    return psi


def psiu_26(z_L: float | NDArrayRealNum) -> NDArrayRealNum:
    """Compute the velocity structure function given z/L.

    :param z_L: stability parameter
    :type z_L: float | NDArrayRealNum
    :return: velocity structure function
    :rtype: NDArrayRealNum
    """
    zet = np.asarray(z_L, dtype=np.float64)
    # compute psi_u for stable conditions by Beljaars & Holtslag 1991
    a = 0.7
    b = 3.0 / 4.0
    c = 5.0
    d = 0.35
    dzet = d * zet
    dzet[dzet > 50] = 50.0
    psi = np.nan * np.empty(zet.shape, dtype=np.float64)
    k = zet >= 0
    psi[k] = -(a * zet[k] + b * (zet[k] - c / d) * np.exp(-dzet[k]) + b * c / d)
    # compute convective psi for unstable conditions by Grachev et. al., 2000
    k = zet < 0  # only compute where zet < 0
    x = (1 - 15 * zet[k]) ** (1 / 4)
    psik = (
        2.0 * np.log((1.0 + x) / 2.0)
        + np.log((1.0 + x * x) / 2.0)  # kansas psi
        - 2.0 * np.arctan(x)
        + np.pi / 2
    )
    x = (1 - 10.15 * zet[k]) ** (1 / 3)
    psic = (
        3 / 2 * np.log((x**2 + x + 1) / 3)  # free convective psi
        - np.sqrt(3) * np.arctan((2 * x + 1) / np.sqrt(3))
        + np.pi / np.sqrt(3)
    )
    # combine free convective and kansas psi
    f = zet[k] ** 2 / (1.0 + zet[k] ** 2)
    psi[k] = (1 - f) * psik + f * psic
    return psi


def psiu_40(z_L: float | NDArrayRealNum) -> NDArrayRealNum:
    """Compute velocity structure function given z/L.

    :param z_L: stability parameter
    :type z_L: float | NDArrayRealNum
    :return: velocity structure function
    :rtype: NDArrayRealNum
    """
    zet = np.asarray(z_L, dtype=np.float64)
    # compute psi_u for stable conditions by Beljaars & Holtslag 1991
    a = 1.0
    b = 3.0 / 4.0
    c = 5.0
    d = 0.35
    dzet = d * zet
    dzet[dzet > 50] = 50.0
    psi = np.nan * np.empty(zet.shape, dtype=np.float64)
    k = zet >= 0
    psi[k] = -(a * zet[k] + b * (zet[k] - c / d) * np.exp(-dzet[k]) + b * c / d)
    # compute convective psi for unstable conditions by Grachev et. al., 2000
    k = np.flatnonzero(zet < 0)
    x = (1.0 - 18.0 * zet[k]) ** (1 / 4)
    psik = (
        2.0 * np.log((1.0 + x) / 2.0)
        + np.log((1.0 + x * x) / 2.0)  # kansas psi
        - 2.0 * np.arctan(x)
        + np.pi / 2
    )
    x = (1.0 - 10 * zet[k]) ** (1 / 3)
    psic = (
        3 / 2 * np.log((x**2 + x + 1) / 3)  # free convective psi
        - np.sqrt(3) * np.arctan((2 * x + 1) / np.sqrt(3))
        + np.pi / np.sqrt(3)
    )
    # combine free convective and kansas psi
    f = zet[k] ** 2 / (1.0 + zet[k] ** 2)
    psi[k] = (1 - f) * psik + f * psic
    return psi


def _check_size(
    arr: float | NDArrayRealNum | xr.DataArray | None,
    N: tuple[int, ...] | int,
    name: str = "Input",
    warn=False,  # noqa: FBT002
) -> NDArrayRealNum[Any, np.dtype[np.float64]]:
    if isinstance(arr, xr.DataArray):
        arr = arr.data
    if arr is None:
        return np.full(N, np.nan, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != N and arr.size != 1:
        msg = f"pyCOARE: {name} array of shape {arr.shape} different shape than u array of shape {N}"
        raise ValueError(msg)
    if arr.size == 1:
        if warn:
            warnings.warn(
                f"pyCOARE: {name} array of length 1, broadcasting to length {N}",
                UserWarning,
                stacklevel=2,
            )
        arr = np.full(N, arr.item(), dtype=np.float64)
        return arr
    return arr


class _output:
    """Base class for all output classes."""

    _metadata = yaml.safe_load(pkgutil.get_data("pycoare", "metadata.yaml"))

    def __init__(self, _bulk_loop_inputs, _bulk_loop_outputs, *args):  # noqa: ARG002, ANN002 - *args necessary for type checker on sub classes
        self._bulk_loop_inputs = _bulk_loop_inputs
        self._bulk_loop_outputs = _bulk_loop_outputs


def _xarray_getters(cls: type[_output]) -> type[_output]:
    """Convert class attributes to xarray DataArrays if the input was an xarray DataArray."""

    def make_getter(name: str) -> Callable[[_output], xr.DataArray | NDArrayRealNum]:
        def getter(self):
            attr = getattr(self, name)
            # in case the attribute is used on an incorrect class
            if not hasattr(self, "_bulk_loop_inputs") or not hasattr(
                self._bulk_loop_inputs,
                "input_type",
            ):
                msg = f"{cls.__name__} instance has no attribute '_bulk_loop_inputs' or 'input_type'"
                raise AttributeError(msg)
            if self._bulk_loop_inputs.input_type == xr.DataArray:
                return xr.DataArray(
                    attr,
                    dims=self._bulk_loop_inputs.dims,
                    coords=self._bulk_loop_inputs.coords,
                    attrs=self._metadata.get(name.lstrip("_"), {}),
                )
            return attr

        return getter

    def make_setter(
        name: str,
    ) -> Callable[[_output, float | NDArrayRealNum | xr.DataArray], None]:
        def setter(self, value):
            setattr(self, name, value)

        return setter

    for field in dir(cls):
        # ensure we only have private fields that start with a single underscore, are not callable, and are not dunder methods
        if (
            field.startswith("_")
            and not field.startswith("__")
            and not callable(getattr(cls, field))
        ):
            # add getter and setter for the public field name, constructed based on the private field name
            setattr(
                cls,
                field.lstrip("_"),
                property(make_getter(field), make_setter(field)),
            )

    return cls
