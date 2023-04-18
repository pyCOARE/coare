"""
Functions for COARE model bulk flux calculations.

Translated and vectorized from J Edson/ C Fairall MATLAB scripts.

Execute '%run coare35vn.py' from the iPython command line for test run with
'test_35_data.txt' input data file.

Byron Blomquist, CU/CIRES, NOAA/ESRL/PSD3
Ludovic Bariteau, CU/CIRES, NOAA/ESRL/PSD3
Andrew Scherer, OSU
v1: May 2015:
Initial Design?
v2: July 2020:
Fixed some typos and changed syntax for python 3.7 compatibility.
v3: April 2022:
Formatted code to be in closer agreement with PEP8,
established testing, packaging.
"""

import numpy as np


def c35(u, t=10, rh=75, zu=10, zt=10, zq=10, ts=10, P=1015, lat=45,
        zi=600, Rs=150, Rl=370, rain=None, cp=None, sigH=None, jcool=1,
        out='full'):
    """
    usage: A = coare35vn(u)  -  include other kwargs as desired

    Vectorized version of COARE 3 code (Fairall et al, 2003) with modification
    based on the CLIMODE, MBL and CBLAST experiments (Edson et al., 2013).
    The cool skin option is retained but warm layer and surface wave options
    have been removed.

    This version includes parameterizations of wave height and wave slope using
    cp and sigH.  Unless these are provided the wind speed dependent
    formulation is used.

    AN IMPORTANT COMPONENT OF THIS CODE IS WHETHER INPUT 'ts' REPRESENTS
    THE SKIN TEMPERATURE OR A NEAR SURFACE TEMPERATURE.  How this variable is
    treated is determined by the jcool parameter:  jcool=1 if Ts is bulk
    ocean temperature (default), jcool=0 if Ts is ocean skin temperature.

    The code assumes u, t, rh, and ts are vectors, but the latter 3 can be
    a constant; rain, if given, is a vector;
    P, Rs, Rl, lat, zi, cp and sigH may be passed as vectors or constants;
    sensor heights (zu, zt, zq) are only constants.  All vectors must be of
    equal length.

    Default values are assigned for all variables except u,t,rh,ts.  Input
    arrays may contain NaNs to indicate missing values.  Defaults should be set
    to representative regional values if possible.

    Input definitions:
    col    var     description
    -------------------------------------------------------------------------
    0      u       ocean surface relative wind speed (m/s) at height zu(m)
    1      t       bulk air temperature (degC) at height zt(m) (default=10)
    2      rh      relative humidity (%) at height zq(m) (default=0.5)
    3      ts      sea water temperature (degC) - see jcool below (default=10)
    4      P       surface air pressure (mb) (default=1015)
    5      Rs      downward shortwave radiation (W/m^2) (default=150)
    6      Rl      downward longwave radiation (W/m^2) (default=370)
    7      zu      wind sensor height (m) (default=18m)
    8      zt      bulk temperature sensor height (m) (default=18m)
    9      zq      RH sensor height (m) (default=18m)
    10     lat     lat = latitude (default=45 N)
    11     zi      zi = PBL height (m) (default=600m)
    12     rain    rain rate (mm/hr) (default=None)
    13     cp      phase speed of dominant waves (m/s)
    14     sigH    significant wave height (m)
    15     jcool   cool skin option (default = 1 for bulk SST)

    Output is a 2-D ndarray with the following variables as 37 columns.
    Other quantities may be added to output by editing lines 536/537.
    col    var     description
    -------------------------------------------------------------------------
    0      usr     friction velocity that includes gustiness (m/s)
    1      tau     wind stress (N/m^2)
    2      hsb     sensible heat flux into ocean (W/m^2)
    3      hlb     latent heat flux into ocean (W/m^2)
    4      hbb     buoyancy flux into ocean (W/m^2)
    5      hsbb    "sonic" buoyancy flux measured directly by sonic anemometer
    6      hlwebb  Webb correction for latent heat flux, add this to directly
                   measured eddy covariance latent heat flux from water vapor
                   mass concentration sensors (e.g. Licor 7500).
    7      tsr     temperature scaling parameter (K)
    8      qsr     specific humidity scaling parameter (g/Kg)
    9      zot     thermal roughness length (m)
    10     zoq     moisture roughness length (m)
    11     Cd      wind stress transfer (drag) coefficient at height zu
    12     Ch      sensible heat transfer coefficient (Stanton number) at ht zu
    13     Ce      latent heat transfer coefficient (Dalton number) at ht zq
    14     L       Obukhov length scale (m)
    15     zet     Monin-Obukhov stability parameter zu/L
    16     dter    cool-skin temperature depression (degC)
    17     dqer    cool-skin humidity depression (degC)
    18     tkt     cool-skin thickness (m)
    19     Urf     wind speed at reference height (select height below)
    20     Trf     temperature at reference height
    21     Qrf     specific humidity at reference height
    22     RHrf    relative humidity at reference height
    23     UrfN    neutral value of wind speed at reference height
    24     Rnl     Upwelling IR radiation computed by COARE
    25     Le      latent heat of vaporization
    26     rhoa    density of air
    27     UN      neutral value of wind speed at zu
    28     U10     wind speed adjusted to 10 m
    29     U10N    neutral value of wind speed at 10m
    30     Cdn_10  neutral value of drag coefficient at 10m
    31     Chn_10  neutral value of Stanton number at 10m
    32     Cen_10  neutral value of Dalton number at 10m
    33     RF      rain heat flux (W/m2)
    34     Evap    evaporation (mm/hr)
    35     Qs      sea surface specific humidity (g/kg)
    36     Q10     specific humidity at 10m (g/kg)
    37     RH10    RH at 10m (%)

    Notes:
    1) u is the ocean-relative wind speed, i.e., the magnitude of the
       difference between the wind (at zu) and ocean surface current
       vectors.
    2) Set jcool=0 if ts is true surface skin temperature,
       otherwise ts is assumed the bulk temperature and jcool=1.
    3) The code to compute the heat flux caused by precipitation is
       included if rain data is available (default is no rain).
    4) Code updates the cool-skin temperature depression dter and thickness
       tkt during iteration loop for consistency.
    5) Number of iterations set to nits = 6.
    6) The warm layer is not implemented in this version.

    Reference:

    Fairall, C.W., E.F. Bradley, J.E. Hare, A.A. Grachev, and J.B. Edson (2003)
    Bulk parameterization of air sea fluxes: updates and verification for the
    COARE algorithm, J. Climate, 16, 571-590.

    Code history:

    1) 12/14/05 - created based on scalar version coare26sn.m with input
       on vectorization from C. Moffat.
    2) 12/21/05 - sign error in psiu_26 corrected, and code added to use
       variable values from the first pass through the iteration loop for the
       stable case with very thin M-O length relative to zu (zetu>50) (as is
       done in the scalar coare26sn and COARE3 codes).
    3) 7/26/11 - S = dt was corrected to read S = ut.
    4) 7/28/11 - modification to roughness length parameterizations based
       on the CLIMODE, MBL, Gasex and CBLAST experiments are incorporated
    5) Python translation by BWB, Oct 2014.  Modified to allow user specified
       vectors for lat and zi.  Defaults added for zu, zt, zq.
    """

    # be sure array inputs are ndarray floats
    # if inputs are already ndarray float this does nothing
    # otherwise copies are created in the local namespace
    u = np.copy(np.asarray(u, dtype=float))
    N = u.size
    # format optional array inputs
    t = check_size(t, N, 't')
    rh = check_size(rh, N, 'rh')
    ts = check_size(ts, N, 'ts')
    P = check_size(P, N, 'P')
    Rs = check_size(Rs, N, 'Rs')
    Rl = check_size(Rl, N, 'Rl')
    zi = check_size(zi, N, 'zi')
    lat = check_size(lat, N, 'lat')
    rain = check_size(rain, N, 'rain')
    zu = check_size(zu, N, 'zu')
    zq = check_size(zq, N, 'zq')
    zt = check_size(zt, N, 'zt')

    if (cp is not None) and (~np.all(np.isnan(cp))):
        waveage_flag = True
        cp = np.copy(np.asarray(cp, dtype=float))
        if cp.size != N:
            raise ValueError(
                'coare35vn: cp array of different length'
            )
        elif cp.size == 1:
            cp = cp * np.ones(N)
    elif (cp is None) or (np.all(np.isnan(cp))):
        waveage_flag = False
        cp = np.nan * np.ones(N)

    if (sigH is not None) and (~np.all(np.isnan(cp))):
        seastate_flag = True
        sigH = np.copy(np.asarray(sigH, dtype=float))
        if sigH.size != N:
            raise ValueError(
                'coare35vn: sigH array of different length'
            )
        elif sigH.size == 1:
            sigH = sigH * np.ones(N)
    elif (sigH is None) or (np.all(np.isnan(cp))):
        seastate_flag = False
        sigH = np.nan * np.ones(N)

    # check jcool
    if jcool != 0:
        jcool = 1   # all input other than 0 defaults to jcool=1

    # input variable u is surface relative wind speed (magnitude of difference
    # between wind and surface current vectors). To follow orginal Fairall
    # code, we set surface current speed us=0. If us data are available
    # construct u prior to using this code.
    us = np.zeros(N)

    # convert rh to specific humidity
    Qs = qsea(ts, P)/1000.0  # surface water specific humidity (kg/kg)
    Q, Pv = qair(t, P, rh)    # specific hum. and partial Pv (mb)
    Q /= 1000.0                   # Q (kg/kg)

    # set constants
    Beta = 1.2
    von = 0.4          # von Karman const
    fdg = 1.00         # Turbulent Prandtl number
    tdk = 273.16
    grav = grv(lat)

    # air constants
    Rgas = 287.1
    Le = (2.501 - 0.00237*ts) * 1e6
    cpa = 1004.67
    rhoa = P*100. / (Rgas * (t + tdk) * (1 + 0.61*Q))
    visa = 1.326e-5 * (1 + 6.542e-3*t + 8.301e-6*t**2 - 4.84e-9*t**3)

    # cool skin constants
    Al = 2.1e-5 * (ts + 3.2)**0.79
    be = 0.026
    cpw = 4000.
    rhow = 1022.
    visw = 1.e-6
    tcw = 0.6
    bigc = 16. * grav * cpw * (rhow * visw)**3 / (tcw**2 * rhoa**2)
    wetc = 0.622 * Le * Qs / (Rgas * (ts + tdk)**2)

    # net radiation fluxes
    Rns = 0.945 * Rs        # albedo correction
    Rnl = 0.97 * (5.67e-8 * (ts - 0.3*jcool + tdk)**4 - Rl)  # initial value

    # BEGIN BULK LOOP

    # first guess
    du = u - us
    dt = ts - t - 0.0098*zt
    dq = Qs - Q
    ta = t + tdk
    ug = 0.5
    dter = 0.3
    ut = np.sqrt(du**2 + ug**2)
    u10 = ut * np.log(10/1e-4) / np.log(zu/1e-4)
    usr = 0.035 * u10
    zo10 = 0.011 * usr**2 / grav + 0.11*visa / usr
    Cd10 = (von / np.log(10/zo10))**2
    Ch10 = 0.00115
    Ct10 = Ch10 / np.sqrt(Cd10)
    zot10 = 10 / np.exp(von/Ct10)
    Cd = (von / np.log(zu/zo10))**2
    Ct = von / np.log(zt/zot10)
    CC = von * Ct/Cd
    Ribcu = -zu / zi / 0.004 / Beta**3
    Ribu = -grav * zu/ta * ((dt - dter*jcool) + 0.61*ta*dq) / ut**2
    zetu = CC * Ribu * (1 + 27/9 * Ribu/CC)

    k50 = find(zetu > 50)   # stable with thin M-O length relative to zu
    k = find(Ribu < 0)

    if Ribcu.size == 1:
        zetu[k] = CC[k] * Ribu[k] / (1 + Ribu[k] / Ribcu)
    else:
        zetu[k] = CC[k] * Ribu[k] / (1 + Ribu[k] / Ribcu[k])

    L10 = zu / zetu
    gf = ut / du
    usr = ut * von / (np.log(zu/zo10) - psiu_40(zu/L10))
    tsr = (-(dt - dter*jcool)*von*fdg
           / (np.log(zt/zot10) - psit_26(zt/L10)))
    qsr = (-(dq - wetc*dter*jcool)*von*fdg
           / (np.log(zq/zot10) - psit_26(zq/L10)))
    tkt = 0.001 * np.ones(N)

    # The following gives the new formulation for the Charnock variable
    charnC = 0.011 * np.ones(N)
    umax = 19
    a1 = 0.0017
    a2 = -0.0050

    charnC = a1 * u10 + a2
    j = find(u10 > umax)
    charnC[j] = a1 * umax + a2

    A = 0.114   # wave-age dependent coefficients
    B = 0.622

    Ad = 0.091  # Sea-state/wave-age dependent coefficients
    Bd = 2.0

    charnW = A * (usr/cp)**B
    zoS = sigH * Ad * (usr/cp)**Bd
    charnS = zoS * grav / usr / usr

    charn = 0.011 * np.ones(N)
    k = find(ut > 10)
    charn[k] = 0.011 + (ut[k] - 10) / (18 - 10)*(0.018 - 0.011)
    k = find(ut > 18)
    charn[k] = 0.018

    # begin bulk loop
    nits = 10   # number of iterations
    for i in range(nits):
        zet = von*grav*zu / ta*(tsr + 0.61*ta*qsr) / (usr**2)
        if waveage_flag:
            if seastate_flag:
                charn = charnS
            else:
                charn = charnW
        else:
            charn = charnC

        L = zu / zet
        zo = charn*usr**2/grav + 0.11*visa/usr  # surface roughness
        rr = zo*usr/visa

        # thermal roughness lengths give Stanton and Dalton numbers that
        # closely approximate COARE 3.0
        zoq = np.minimum(1.6e-4, 5.8e-5/rr**0.72)
        zot = zoq
        cdhf = von / (np.log(zu/zo) - psiu_26(zu/L))
        cqhf = von*fdg / (np.log(zq/zoq) - psit_26(zq/L))
        cthf = von*fdg / (np.log(zt/zot) - psit_26(zt/L))

        usr = ut*cdhf
        qsr = -(dq - wetc*dter*jcool)*cqhf
        tsr = -(dt - dter*jcool)*cthf
        tvsr = tsr + 0.61*ta*qsr
        Bf = -grav / ta*usr*tvsr
        ug = 0.2 * np.ones(N)
        k = find(Bf > 0)

        if zi.size == 1:
            ug[k] = Beta*(Bf[k]*zi)**0.333
        else:
            ug[k] = Beta*(Bf[k]*zi[k])**0.333

        ut = np.sqrt(du**2 + ug**2)
        gf = ut/du
        hsb = -rhoa*cpa*usr*tsr
        hlb = -rhoa*Le*usr*qsr
        qout = Rnl + hsb + hlb
        # use tkt=0.001
        dels = Rns * (0.065 + 11*tkt - 6.6e-5/tkt*(1 - np.exp(-tkt/8.0e-4)))
        qcol = qout - dels
        alq = Al*qcol + be*hlb*cpw/Le

        xlamx = 6.0 * np.ones(N)
        # redefine tkt
        tkt = np.minimum(0.01, xlamx*visw/(np.sqrt(rhoa/rhow)*usr))
        k = find(alq > 0)
        xlamx[k] = 6/(1 + (bigc[k]*alq[k]/usr[k]**4)**0.75)**0.333
        tkt[k] = xlamx[k]*visw / (np.sqrt(rhoa[k]/rhow)*usr[k])
        dter = qcol*tkt/tcw
        dqer = wetc*dter
        Rnl = 0.97*(5.67e-8*(ts - dter*jcool + tdk)**4 - Rl)   # update dter

        # save first iteration solution for case of zetu>50
        if i == 0:
            usr50 = usr[k50]
            tsr50 = tsr[k50]
            qsr50 = qsr[k50]
            L50 = L[k50]
            zet50 = zet[k50]
            dter50 = dter[k50]
            dqer50 = dqer[k50]
            tkt50 = tkt[k50]

        u10N = usr/von/gf*np.log(10/zo)
        charnC = a1*u10N + a2
        k = find(u10N > umax)
        charnC[k] = a1*umax + a2
        charnW = A*(usr/cp)**B
        zoS = sigH*Ad*(usr/cp)**Bd - 0.11*visa/usr
        charnS = zoS*grav/usr/usr

    # end bulk loop

    # insert first iteration solution for case with zetu > 50
    usr[k50] = usr50
    tsr[k50] = tsr50
    qsr[k50] = qsr50
    L[k50] = L50
    zet[k50] = zet50
    dter[k50] = dter50
    dqer[k50] = dqer50
    tkt[k50] = tkt50

    # compute fluxes
    tau = rhoa*usr*usr/gf           # wind stress
    hsb = -rhoa*cpa*usr*tsr         # sensible heat flux
    hlb = -rhoa*Le*usr*qsr          # latent heat flux
    wbar = 1.61*hlb/Le/(1+1.61*Q)/rhoa + hsb/rhoa/cpa/ta
    hlwebb = rhoa*wbar*Q*Le

    # compute transfer coeffs relative to ut @ meas. ht
    Cd = tau/rhoa/ut/np.maximum(0.1, du)
    Ch = -usr*tsr/ut/(dt - dter*jcool)
    Ce = -usr*qsr/(dq - dqer*jcool)/ut

    # compute 10-m neutral coeff relative to ut
    Cdn_10 = 1000*von**2 / np.log(10/zo)**2
    Chn_10 = 1000*von**2 * fdg/np.log(10/zo) / np.log(10/zot)
    Cen_10 = 1000*von**2 * fdg/np.log(10/zo) / np.log(10/zoq)

    # compute the stability functions
    psi = psiu_26(zu/L)
    psi10 = psiu_26(10/L)
    gf = ut/du

    # Determine the wind speeds relative to ocean surface
    # Note that usr is the friction velocity that includes
    # gustiness usr = sqrt(Cd) S, which is equation (18) in
    # Fairall et al. (1996)
    S = ut
    S10 = S + usr/von*(np.log(10/zu) - psi10 + psi)
    U10 = S10/gf

    # rain heat flux after Gosnell et al., JGR, 1995
    if rain is None:
        RF = np.zeros(usr.size)
    else:
        # water vapour diffusivity
        dwat = 2.11e-5*((t + tdk)/tdk)**1.94

        # heat diffusivity
        dtmp = (1 + 3.309e-3*t - 1.44e-6*t**2) * 0.02411/(rhoa*cpa)

        # Clausius-Clapeyron
        dqs_dt = Q*Le / (Rgas*(t + tdk)**2)

        # wet bulb factor
        alfac = 1/(1 + 0.622*(dqs_dt*Le*dwat)/(cpa*dtmp))
        RF = rain*alfac*cpw*((ts-t-dter*jcool)+(Qs-Q-dqer*jcool)*Le/cpa)/3600

    dqer = wetc*dter*jcool
    SSQ = Qs - dqer
    SSQ = SSQ*1000

    Q = Q*1000
    qsr = qsr*1000

    if out == 'full':
        A = np.squeeze(
            np.column_stack(
                np.array([
                    usr, tau, hsb, hlb, hlwebb, tsr, qsr,
                    zot, zoq, Cd, Ch, Ce, L, zet, dter, dqer,
                    tkt, RF, Cdn_10, Chn_10, Cen_10
                ])
            )
        )
    elif out == 'u10':
        A = np.squeeze(np.column_stack(np.array([U10])))

    elif out == 'tau':
        A = np.squeeze(np.column_stack(np.array([tau])))

    return A


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
