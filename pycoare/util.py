import numpy as np
from numpy.typing import ArrayLike, NDArray


def grv(lat: ArrayLike) -> NDArray[np.float64]:
    """Normal gravity at latitude lat (degrees) using the WGS84 ellipsoid.

    :param lat: latitude (degrees)
    :type lat: ArrayLike
    :return: normal gravity (m/s^2)
    :rtype: NDArray[np.float64]
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
    t: ArrayLike,
    p: ArrayLike,
    q: ArrayLike,
) -> NDArray[np.float64]:
    """Compute relative humidity from temperature, pressure, and specific humidity.

    :param t: temperature (degC)
    :type t: ArrayLike
    :param p: pressure (mb)
    :type p: ArrayLike
    :param q: specific humidity (g/kg)
    :type q: ArrayLike
    :return: relative humidity (%)
    :rtype: NDArray[np.float64]
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    es = qsat(t, p)
    em = p * q / (0.622 + 0.378 * q)
    rh = 100.0 * em / es
    return rh


def qsat(t: ArrayLike, p: ArrayLike) -> NDArray[np.float64]:
    """Returns saturation vapor pressure from temperature and pressure.

    :param t: temperature (degC)
    :type t: ArrayLike
    :param p: pressure (mb)
    :type p: ArrayLike
    :return: saturation vapor pressure (g/kg)
    :rtype: NDArray[np.float64]
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    es = 6.1121 * np.exp(17.502 * t / (240.97 + t))
    es = es * (1.0007 + p * 3.46e-6)
    return es


def qsea(t: ArrayLike, p: ArrayLike, s: ArrayLike = 35) -> NDArray[np.float64]:
    """Returns saturation specific humidity at sea surface from temperature and pressure.

    :param t: temperature (degC)
    :type t: ArrayLike
    :param p: pressure (mb)
    :type p: ArrayLike
    :return: saturation specific humidity (g/kg)
    :rtype: NDArray[np.float64]
    """
    ex = qsat(t, p)  # returns ex as ndarray float
    es = (1 - 0.02 * s / 35) * ex
    qs = 622 * es / (p - 0.378 * es)
    return qs


def qair(t: ArrayLike, p: ArrayLike, rh: ArrayLike) -> NDArray[np.float64]:
    """Returns specific humidity given temperature, pressure, and relative humidity.

    :param t: temperature (degC)
    :type t: ArrayLike
    :param p: pressure (mb)
    :type p: ArrayLike
    :param rh: relative humidity (%)
    :type rh: ArrayLike
    :return: specific humidity (g/kg), partial pressure (mb)
    :rtype: tuple[NDArray[np.float64], NDArray[np.float64]]
    """
    rh = np.asarray(rh, dtype=float)
    rh /= 100.0
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=float)
    em = rh * qsat(t, p=p)
    qa = 621.97 * em / (p - 0.378 * em)
    return qa


def psit_26(z_L: ArrayLike) -> NDArray[np.float64]:
    """Computes the temperature structure function given z/L

    :param z_L: stability parameter
    :type z_L: ArrayLike
    :return: temperature structure function
    :rtype: NDArray[np.float64]
    """
    zet = np.asarray(z_L, dtype=float)
    # compute psi_t for stable conditions by Beljaars & Holtslag 1991
    a = 1
    b = 0.6667
    c = 5
    d = 0.35
    dzet = d * zet
    dzet[dzet > 50] = 50.0
    psi = np.nan * np.empty(zet.shape, dtype=float)
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


def psiu_26(z_L: ArrayLike) -> NDArray[np.float64]:
    """Computes the velocity structure function given z/L

    :param z_L: stability parameter
    :type z_L: ArrayLike
    :return: velocity structure function
    :rtype: NDArray[np.float64]
    """
    zet = np.asarray(z_L, dtype=float)
    # compute psi_u for stable conditions by Beljaars & Holtslag 1991
    a = 0.7
    b = 3.0 / 4.0
    c = 5.0
    d = 0.35
    dzet = d * zet
    dzet[dzet > 50] = 50.0
    psi = np.nan * np.empty(zet.shape, dtype=float)
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


def psiu_40(z_L: ArrayLike) -> NDArray[np.float64]:
    """Computes velocity structure function given z/L

    :param z_L: stability parameter
    :type z_L: ArrayLike
    :return: velocity structure function
    :rtype: NDArray[np.float64]
    """
    zet = np.asarray(z_L, dtype=float)
    # compute psi_u for stable conditions by Beljaars & Holtslag 1991
    a = 1.0
    b = 3.0 / 4.0
    c = 5.0
    d = 0.35
    dzet = d * zet
    dzet[dzet > 50] = 50.0
    psi = np.nan * np.empty(zet.shape, dtype=float)
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
    arr: ArrayLike,
    N: int,
    name: str = "Input",
    warn=False,
) -> NDArray[np.float64]:
    arr = np.asarray(arr, dtype=float)
    if arr.shape != N and arr.size != 1:
        raise ValueError(
            f"pyCOARE: {name} array of shape {arr.shape} different shape than u array of shape {N}",
        )
    if arr.size == 1:
        if warn:
            print(f"pyCOARE: {name} array of length 1, broadcasting to length {N}")
        arr = arr * np.ones(N, dtype=np.float64)
        return arr
    return arr
