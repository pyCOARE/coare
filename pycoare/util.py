import numpy as np


def grv(lat):
    # computes g [m/sec^2] given lat in deg
    gamma = 9.7803267715
    c1 = 0.0052790414
    c2 = 2.32718e-05
    c3 = 1.262e-07
    c4 = 7e-10
    phi = lat * np.pi / 180
    x = np.sin(phi)
    g = gamma * (1 + c1 * x ** 2 + c2 * x ** 4 + c3 * x ** 6 + c4 * x ** 8)
    return g


def rhcalc(t, p, q):
    """
    usage: rh = rhcalc(t,p,q)
    Returns RH(%) for given t(C), p(mb) and specific humidity, q(kg/kg)

    Returns ndarray float for any numeric object input.
    """
    q2 = np.copy(np.asarray(q, dtype=float))    # conversion to ndarray float
    p2 = np.copy(np.asarray(p, dtype=float))
    t2 = np.copy(np.asarray(t, dtype=float))
    es = qsat(t2, p2)
    em = p2 * q2 / (0.622 + 0.378 * q2)
    rh = 100.0 * em / es
    return rh


def find(b):
    """

    Usage: idx = find(b) - Returns sorted array of indices where boolean
    input array b is true.  Similar to MATLAB find function.

    Input may be a 1-D boolean array or any expression that evaluates to
    a 1-D boolean: e.g. ii = find(x < 3), where x is a 1-D ndarray.
    This syntax is similar to MATLAB usage.

    2-D or higher arrays could be flattened prior to calling find() and
    then reconstituted with reshape.  This modification could be added to
    this function as well to support N-D arrays.

    Returns 1-D ndarray of int64.

    """
    if type(b) != np.ndarray:
        raise ValueError('find: Input should be ndarray')
    if b.dtype != 'bool':
        raise ValueError('find: Input should be boolean array')
    if b.ndim > 1:
        raise ValueError('find: Input should be 1-D')

    F = b.size - np.sum(b)    # number of False in b
    idx = np.argsort(b)[F:]   # argsort puts True at the end, so select [F:]
    idx = np.sort(idx)        # be sure values in idx are ordered low to high

    return idx


def qsat(t, p):
    """
    usage: es = qsat(t,p)
    Returns saturation vapor pressure es (mb) given t(C) and p(mb).

    After Buck, 1981: J.Appl.Meteor., 20, 1527-1532

    Returns ndarray float for any numeric object input.
    """
    t2 = np.copy(np.asarray(t, dtype=float))  # convert to ndarray float
    p2 = np.copy(np.asarray(p, dtype=float))
    es = 6.1121 * np.exp(17.502 * t2 / (240.97 + t2))
    es = es * (1.0007 + p2 * 3.46e-6)
    return es


def qsea(t, p):
    """
    usage: qs = qsea(t,p)
    Returns saturation specific humidity (g/kg) at sea surface
    given t(C) and p(mb) input of any numeric type.

    Returns ndarray float for any numeric object input.
    """
    ex = qsat(t, p)  # returns ex as ndarray float
    es = 0.98 * ex
    qs = 622 * es / (p - 0.378 * es)
    return qs


def qair(t, p, rh):
    """
    usage: qa, em = qair(t,p,rh)
    Returns specific humidity (g/kg) and partial pressure (mb)
    given t(C), p(mb) and rh(%).

    Returns ndarray float for any numeric object input.
    """
    rh2 = np.copy(np.asarray(rh, dtype=float))  # conversion to ndarray float
    rh2 /= 100.0                         # frational rh
    p2 = np.copy(np.asarray(p, dtype=float))
    t2 = np.copy(np.asarray(t, dtype=float))
    em = rh2 * qsat(t2, p2)
    qa = 621.97 * em / (p2 - 0.378 * em)
    return (qa, em)


def psit_26(z_L):
    """
    usage psi = psit_26(z_L)

    Computes the temperature structure function given z/L.
    """
    zet = np.copy(np.asarray(z_L, dtype=float))  # conversion to ndarray float
    dzet = 0.35*zet
    dzet[dzet > 50] = 50.           # stable
    psi = -((1 + 0.6667*zet)**1.5 + 0.6667*(zet - 14.28)*np.exp(-dzet) + 8.525)
    k = find(zet < 0)            # unstable
    x = (1 - 15*zet[k])**0.5
    psik = 2*np.log((1 + x)/2.)
    x = (1 - 34.15*zet[k])**0.3333
    psic = (1.5*np.log((1.+x+x**2)/3.)
            - np.sqrt(3)*np.arctan((1 + 2*x)/np.sqrt(3)))
    psic += 4*np.arctan(1.)/np.sqrt(3.)
    f = zet[k]**2 / (1. + zet[k]**2.)
    psi[k] = (1-f)*psik + f*psic
    return psi


def psiu_26(z_L):
    """
    usage: psi = psiu_26(z_L)

    Computes velocity structure function given z/L
    """
    zet = np.copy(np.asarray(z_L, dtype=float))   # conversion to ndarray float
    dzet = 0.35*zet
    dzet[dzet > 50] = 50.           # stable
    a = 0.7
    b = 3./4.
    c = 5.
    d = 0.35
    psi = -(a*zet + b*(zet - c/d)*np.exp(-dzet) + b*c/d)
    k = find(zet < 0)         # unstable
    x = (1 - 15*zet[k])**0.25
    psik = (2.*np.log((1.+x)/2.) + np.log((1.+x*x)/2.)
            - 2.*np.arctan(x) + 2.*np.arctan(1.))
    x = (1 - 10.15*zet[k])**0.3333
    psic = (1.5*np.log((1.+x+x**2)/3.)
            - np.sqrt(3.)*np.arctan((1.+2.*x)/np.sqrt(3.)))
    psic += 4*np.arctan(1.)/np.sqrt(3.)
    f = zet[k]**2 / (1.+zet[k]**2)
    psi[k] = (1-f)*psik + f*psic
    return psi


def psiu_40(z_L):
    """
    usage: psi = psiu_40(z_L)

    Computes velocity structure function given z/L
    """
    zet = np.copy(np.asarray(z_L, dtype=float))  # conversion to ndarray float
    dzet = 0.35*zet
    dzet[dzet > 50] = 50.           # stable
    a = 1.
    b = 3./4.
    c = 5.
    d = 0.35
    psi = -(a*zet + b*(zet - c/d)*np.exp(-dzet) + b*c/d)
    k = find(zet < 0)         # unstable
    x = (1. - 18.*zet[k])**0.25
    psik = (2.*np.log((1.+x)/2.) + np.log((1.+x*x)/2.)
            - 2.*np.arctan(x) + 2.*np.arctan(1.))
    x = (1. - 10.*zet[k])**0.3333
    psic = (1.5*np.log((1.+x+x**2)/3.)
            - np.sqrt(3.)*np.arctan((1.+2.*x)/np.sqrt(3.)))
    psic += 4.*np.arctan(1.)/np.sqrt(3.)
    f = zet[k]**2 / (1.+zet[k]**2)
    psi[k] = (1-f)*psik + f*psic
    return psi


def check_size(arr, N, name='Input'):
    arr = np.copy(np.asarray(arr, dtype=float))
    if arr.size != N and arr.size != 1:
        raise ValueError(
            f'coare35vn: {name} array of different length than u'
        )
    elif arr.size == 1:
        arr = arr * np.ones(N)
        return arr
    else:
        return arr


def __return_vars(out, usr, tau, hsb, hlb, hbb, hsbb, hlwebb, tsr, qsr,
                  zot, zoq, Cd, Ch, Ce, L, zet, dter, dqer, tkt, Urf,
                  Trf, Qrf, RHrf, UrfN, Rnl, Le, rhoa, UN, U10, U10N,
                  RF, Cdn_10, Chn_10, Cen_10, Evap, Qs, Q10, RH10):
    if out == 'default':
        A = np.squeeze(
            np.column_stack(
                np.array([
                    usr, tau, hsb, hlb, hlwebb, tsr, qsr,
                    zot, zoq, Cd, Ch, Ce, L, zet, dter, dqer,
                    tkt, RF, Cdn_10, Chn_10, Cen_10
                ])
            )
        )
    elif out == 'all':
        A = np.squeeze(
            np.column_stack(
                np.array([
                    usr, tau, hsb, hlb, hbb, hsbb, hlwebb, tsr, qsr,
                    zot, zoq, Cd, Ch, Ce, L, zet, dter, dqer, tkt, Urf,
                    Trf, Qrf, RHrf, UrfN, Rnl, Le, rhoa, UN, U10, U10N,
                    RF, Cdn_10, Chn_10, Cen_10, Evap, Qs, Q10, RH10
                ])
            )
        )
    elif out == 'usr':
        A = np.squeeze(np.column_stack(np.array([usr])))
    elif out == 'tau':
        A = np.squeeze(np.column_stack(np.array([tau])))
    elif out == 'hsb':
        A = np.squeeze(np.column_stack(np.array([hsb])))
    elif out == 'hlb':
        A = np.squeeze(np.column_stack(np.array([hlb])))
    elif out == 'hbb':
        A = np.squeeze(np.column_stack(np.array([hbb])))
    elif out == 'hlwebb':
        A = np.squeeze(np.column_stack(np.array([hlwebb])))
    elif out == 'tsr':
        A = np.squeeze(np.column_stack(np.array([tsr])))
    elif out == 'qsr':
        A = np.squeeze(np.column_stack(np.array([qsr])))
    elif out == 'zot':
        A = np.squeeze(np.column_stack(np.array([zot])))
    elif out == 'zoq':
        A = np.squeeze(np.column_stack(np.array([zoq])))
    elif out == 'Cd':
        A = np.squeeze(np.column_stack(np.array([Cd])))
    elif out == 'Ch':
        A = np.squeeze(np.column_stack(np.array([Ch])))
    elif out == 'Ce':
        A = np.squeeze(np.column_stack(np.array([Ce])))
    elif out == 'L':
        A = np.squeeze(np.column_stack(np.array([L])))
    elif out == 'zet':
        A = np.squeeze(np.column_stack(np.array([zet])))
    elif out == 'dter':
        A = np.squeeze(np.column_stack(np.array([dter])))
    elif out == 'dqer':
        A = np.squeeze(np.column_stack(np.array([dqer])))
    elif out == 'tkt':
        A = np.squeeze(np.column_stack(np.array([tkt])))
    elif out == 'Urf':
        A = np.squeeze(np.column_stack(np.array([Urf])))
    elif out == 'Trf':
        A = np.squeeze(np.column_stack(np.array([Trf])))
    elif out == 'Qrf':
        A = np.squeeze(np.column_stack(np.array([Qrf])))
    elif out == 'RHrf':
        A = np.squeeze(np.column_stack(np.array([RHrf])))
    elif out == 'UrfN':
        A = np.squeeze(np.column_stack(np.array([UrfN])))
    elif out == 'Rnl':
        A = np.squeeze(np.column_stack(np.array([Rnl])))
    elif out == 'Le':
        A = np.squeeze(np.column_stack(np.array([Le])))
    elif out == 'rhoa':
        A = np.squeeze(np.column_stack(np.array([rhoa])))
    elif out == 'UN':
        A = np.squeeze(np.column_stack(np.array([UN])))
    elif out == 'U10':
        A = np.squeeze(np.column_stack(np.array([U10])))
    elif out == 'U10N':
        A = np.squeeze(np.column_stack(np.array([U10N])))
    elif out == 'Cdn_10':
        A = np.squeeze(np.column_stack(np.array([Cdn_10])))
    elif out == 'Chn_10':
        A = np.squeeze(np.column_stack(np.array([Chn_10])))
    elif out == 'Cen_10':
        A = np.squeeze(np.column_stack(np.array([Cen_10])))
    elif out == 'RF':
        A = np.squeeze(np.column_stack(np.array([RF])))
    elif out == 'Evap':
        A = np.squeeze(np.column_stack(np.array([Evap])))
    elif out == 'Qs':
        A = np.squeeze(np.column_stack(np.array([Qs])))
    elif out == 'Q10':
        A = np.squeeze(np.column_stack(np.array([Q10])))
    elif out == 'RH10':
        A = np.squeeze(np.column_stack(np.array([RH10])))
    else:
        raise ValueError(f'String \'{out}\' is not a valid output variable name! See '
                         + 'https://github.com/pyCOARE/coare/blob/main/docs/io_info/c35_outputs.md'
                         + ' for a list of valid output variable names.')
    return A
