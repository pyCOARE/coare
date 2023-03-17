## TODO 
## limit outputs (optional additional parameter?)
## standardize inputs (what can be len-1 array? presets?)
## 

"""
Functions for COARE model bulk flux calculations.

Translated and vectorized from J Edson/ C Fairall MATLAB scripts.

Execute '%run coare35vn.py' from the iPython command line for test run with
'test_35_data.txt' input data file.

Byron Blomquist, CU/CIRES, NOAA/ESRL/PSD3
Ludovic Bariteau, CU/CIRES, NOAA/ESRL/PSD3
v1: May 2015
v2: July 2020. Fixed some typos and changed syntax for python 3.7 compatibility.
"""

import numpy as np

def c35(u, t=10, rh=0.5, ts=10, P=1015, Rs=150, Rl=370, zu=18, zt=18, zq=18,
        lat=45, zi=600, rain=None, cp=None, sigH=None, jcool=1, out='min'):
    """
    usage: A = coare35vn(u)  -  include other kwargs as desired

    Vectorized version of COARE 3 code (Fairall et al, 2003) with modification
    based on the CLIMODE, MBL and CBLAST experiments (Edson et al., 2013).
    The cool skin option is retained but warm layer and surface wave options
    have been removed.

    This version includes parameterizations of wave height and wave slope using
    cp and sigH.  Unless these are provided the wind speed dependent formulation
    is used.

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
    19     Urf     wind speed at reference height (user can select height below)
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

    Fairall, C.W., E.F. Bradley, J.E. Hare, A.A. Grachev, and J.B. Edson (2003),
    Bulk parameterization of air sea fluxes: updates and verification for the
    COARE algorithm, J. Climate, 16, 571-590.

    Code history:

    1) 12/14/05 - created based on scalar version coare26sn.m with input
       on vectorization from C. Moffat.
    2) 12/21/05 - sign error in psiu_26_3p5 corrected, and code added to use
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

    # these default to 1 element arrays
    t = np.copy(np.asarray(t, dtype=float))
    rh = np.copy(np.asarray(rh, dtype=float))
    ts = np.copy(np.asarray(ts, dtype=float))
    P = np.copy(np.asarray(P, dtype=float))
    Rs = np.copy(np.asarray(Rs, dtype=float))
    Rl = np.copy(np.asarray(Rl, dtype=float))
    zi = np.copy(np.asarray(zi, dtype=float))
    lat = np.copy(np.asarray(lat, dtype=float))

    # check for mandatory input variable consistency
    len = u.size
    if not np.all([t.size==len, rh.size==len, ts.size==len]):
        raise ValueError ('coare35vn: u, t, rh, ts arrays of different length')

    # format optional array inputs
    if P.size != len and P.size != 1:
        raise ValueError ('coare35vn: P array of different length')
    elif P.size == 1:
        P = P * np.ones(len)

    if Rl.size != len and Rl.size != 1:
        raise ValueError ('coare35vn: Rl array of different length')
    elif Rl.size == 1:
        Rl = Rl * np.ones(len)

    if Rs.size != len and Rs.size != 1:
        raise ValueError ('coare35vn: Rs array of different length')
    elif Rs.size == 1:
        Rs = Rs * np.ones(len)

    if zi.size != len and zi.size != 1:
        raise ValueError ('coare35vn: zi array of different length')
    elif zi.size == 1:
        zi = zi * np.ones(len)

    if lat.size != len and lat.size != 1:
        raise ValueError ('coare35vn: lat array of different length')
    elif lat.size == 1:
        lat = lat * np.ones(len)

    if rain is not None:
        rain = np.asarray(rain, dtype=float)
        if rain.size != len:
            raise ValueError ('coare35vn: rain array of different length')
        
    if cp is not None:
        waveage_flag = True
        cp = np.copy(np.asarray(cp, dtype=float))
        if cp.size != len:
            raise ValueError ('coare35vn: cp array of different length')
        elif cp.size == 1:
            cp = cp * np.ones(len)
    else:
        waveage_flag = False
        cp = np.nan * np.ones(len)

    if sigH is not None:
        seastate_flag = True
        sigH = np.copy(np.asarray(sigH, dtype=float))
        if sigH.size != len:
            raise ValueError ('coare35vn: sigH array of different length')
        elif sigH.size == 1:
            sigH = sigH * np.ones(len)
    else:
        seastate_flag = False
        sigH = np.nan * np.ones(len)

    if waveage_flag and seastate_flag:
        print ('Using seastate dependent parameterization')

    if waveage_flag and not seastate_flag:
        print ('Using waveage dependent parameterization')

    # check jcool
    if jcool != 0:
        jcool = 1   # all input other than 0 defaults to jcool=1

    # check sensor heights
    test = [type(zu) is int or type(zu) is float]
    test.append(type(zt) is int or type(zt) is float)
    test.append(type(zq) is int or type(zq) is float)

    if not np.all(test):
        raise ValueError ('coare35vn: zu, zt, zq, should be constants')

    zu = zu * np.ones(len)
    zt = zt * np.ones(len)
    zq = zq * np.ones(len)

    # input variable u is surface relative wind speed (magnitude of difference
    # between wind and surface current vectors). To follow orginal Fairall
    # code, we set surface current speed us=0. If us data are available
    # construct u prior to using this code.
    us = np.zeros(len)

    # convert rh to specific humidity
    Qs = qsea_3p5(ts,P)/1000.0  # surface water specific humidity (kg/kg)
    Q, Pv = qair_3p5(t,P,rh)    # specific hum. and partial Pv (mb)
    Q /= 1000.0                   # Q (kg/kg)

    # set constants
    zref = 10.          # ref height, m (edit as desired)
    Beta = 1.2
    von  = 0.4          # von Karman const
    fdg  = 1.00         # Turbulent Prandtl number
    tdk  = 273.16
    grav = grv(lat)

    # air constants
    Rgas = 287.1
    Le   = (2.501 - 0.00237*ts) * 1e6
    cpa  = 1004.67
    cpv  = cpa * (1 + 0.84*Q)
    rhoa = P*100. / (Rgas * (t + tdk) * (1 + 0.61*Q))
    rhodry = (P - Pv)*100. / (Rgas * (t + tdk))
    visa = 1.326e-5 * (1 + 6.542e-3*t + 8.301e-6*t**2 - 4.84e-9*t**3)

    # cool skin constants
    Al   = 2.1e-5 * (ts + 3.2)**0.79
    be   = 0.026
    cpw  = 4000.
    rhow = 1022.
    visw = 1.e-6
    tcw  = 0.6
    bigc = 16. * grav * cpw * (rhow * visw)**3 / (tcw**2 * rhoa**2)
    wetc = 0.622 * Le * Qs / (Rgas * (ts + tdk)**2)

    # net radiation fluxes
    Rns = 0.945 * Rs        #albedo correction
    Rnl = 0.97 * (5.67e-8 * (ts - 0.3*jcool + tdk)**4 - Rl) # initial value

    #####     BEGIN BULK LOOP

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
    zetu = CC * Ribu * (1 + 27/9  * Ribu/CC)

    k50 = find(zetu > 50)   # stable with thin M-O length relative to zu
    k = find(Ribu < 0)

    if Ribcu.size == 1:
        zetu[k] = CC[k] * Ribu[k] / (1 + Ribu[k] / Ribcu)
    else:
        zetu[k] = CC[k] * Ribu[k] / (1 + Ribu[k] / Ribcu[k])

    L10 = zu / zetu
    gf = ut / du
    usr = ut * von / (np.log(zu/zo10) - psiu_40_3p5(zu/L10))
    tsr = (-(dt - dter*jcool)*von*fdg 
           / (np.log(zt/zot10) - psit_26_3p5(zt/L10)))
    qsr = (-(dq - wetc*dter*jcool)*von*fdg 
           / (np.log(zq/zot10) - psit_26_3p5(zq/L10)))
    tkt = 0.001 * np.ones(len)

    # The following gives the new formulation for the Charnock variable
    charnC = 0.011 * np.ones(len)
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

    charn = 0.011 * np.ones(len)
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
        cdhf = von / (np.log(zu/zo) - psiu_26_3p5(zu/L))
        cqhf = von*fdg / (np.log(zq/zoq) - psit_26_3p5(zq/L))
        cthf = von*fdg / (np.log(zt/zot) - psit_26_3p5(zt/L))

        usr = ut*cdhf
        qsr = -(dq - wetc*dter*jcool)*cqhf
        tsr = -(dt - dter*jcool)*cthf
        tvsr = tsr + 0.61*ta*qsr
        tssr = tsr + 0.51*ta*qsr
        Bf = -grav / ta*usr*tvsr
        ug = 0.2 * np.ones(len)
        k = find(Bf > 0)

        if zi.size == 1:
            ug[k] = Beta*(Bf[k]*zi)**0.333
        else:
            ug[k] = Beta*(Bf[k]*zi[k])**0.333

        ut = np.sqrt(du**2  + ug**2)
        gf = ut/du
        hsb = -rhoa*cpa*usr*tsr
        hlb = -rhoa*Le*usr*qsr
        qout = Rnl + hsb + hlb
        dels = Rns * (0.065 + 11*tkt - 6.6e-5/tkt*(1 - np.exp(-tkt/8.0e-4))) #use tkt=0.001
        qcol = qout - dels
        alq = Al*qcol + be*hlb*cpw/Le
        
        xlamx = 6.0 * np.ones(len)        
        tkt = np.minimum(0.01, xlamx*visw/(np.sqrt(rhoa/rhow)*usr)) #redefine tkt
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
    hbb = -rhoa*cpa*usr*tvsr        # buoyancy flux
    hsbb = -rhoa*cpa*usr*tssr       # sonic buoyancy flux
    wbar = 1.61*hlb/Le/(1+1.61*Q)/rhoa + hsb/rhoa/cpa/ta
    hlwebb = rhoa*wbar*Q*Le
    Evap = 1000*hlb/Le/1000*3600 # mm/hour

    # compute transfer coeffs relative to ut @ meas. ht
    Cd = tau/rhoa/ut/np.maximum(0.1,du)
    Ch = -usr*tsr/ut/(dt - dter*jcool)
    Ce = -usr*qsr/(dq - dqer*jcool)/ut

    # compute 10-m neutral coeff relative to ut
    Cdn_10 = 1000*von**2 / np.log(10/zo)**2
    Chn_10 = 1000*von**2 * fdg/np.log(10/zo) / np.log(10/zot)
    Cen_10 = 1000*von**2 * fdg/np.log(10/zo) / np.log(10/zoq)

    # compute the stability functions
    zrf_u = 10      # User defined reference heights
    zrf_t = 10
    zrf_q = 10

    psi = psiu_26_3p5(zu/L)
    psi10 = psiu_26_3p5(10/L)
    psirf = psiu_26_3p5(zrf_u/L)
    psiT = psit_26_3p5(zt/L)
    psi10T = psit_26_3p5(10/L)
    psirfT = psit_26_3p5(zrf_t/L)
    psirfQ = psit_26_3p5(zrf_q/L)
    gf = ut/du

    # Determine the wind speeds relative to ocean surface
    # Note that usr is the friction velocity that includes
    # gustiness usr = sqrt(Cd) S, which is equation (18) in
    # Fairall et al. (1996)
    S = ut
    U = du
    S10 = S + usr/von*(np.log(10/zu) - psi10 + psi)
    U10 = S10/gf
    Urf = U + usr/von/gf*(np.log(zrf_u/zu) - psirf + psi)
    UN = U + psi*usr/von/gf
    U10N = U10 + psi10*usr/von/gf
    UrfN = Urf + psirf*usr/von/gf
    UN2 = usr/von/gf * np.log(zu/zo)
    U10N2 = usr/von/gf * np.log(10/zo)
    UrfN2 = usr/von/gf * np.log(zrf_u/zo)

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



    lapse = grav/cpa
    SST = ts - dter*jcool
    T = t
    T10 = T + tsr/von*(np.log(10/zt) - psi10T + psiT) + lapse*(zt - 10)
    Trf = T + tsr/von*(np.log(zrf_t/zt) - psirfT + psiT) + lapse*(zt - zrf_t)
    TN = T + psiT*tsr/von
    T10N = T10 + psi10T*tsr/von
    TrfN = Trf + psirfT*tsr/von
    TN2 = SST + tsr/von * np.log(zt/zot) - lapse*zt
    T10N2 = SST + tsr/von * np.log(10/zot) - lapse*10;
    TrfN2 = SST + tsr/von * np.log(zrf_t/zot) - lapse*zrf_t

    dqer = wetc*dter*jcool
    SSQ = Qs - dqer
    SSQ = SSQ*1000

    Q = Q*1000
    qsr = qsr*1000
    Q10 = Q + qsr/von*(np.log(10/zq) - psi10T + psiT)
    Qrf = Q + qsr/von*(np.log(zrf_q/zq) - psirfQ + psiT)
    QN = Q + psiT*qsr/von/np.sqrt(gf)
    Q10N = Q10 + psi10T*qsr/von
    QrfN = Qrf + psirfQ*qsr/von
    QN2 = SSQ + qsr/von * np.log(zq/zoq)
    Q10N2 = SSQ + qsr/von * np.log(10/zoq)
    QrfN2 = SSQ + qsr/von * np.log(zrf_q/zoq)

    RHrf = rhcalc_3p5(Trf, P, Qrf/1000)
    RH10 = rhcalc_3p5(T10, P, Q10/1000)

    if out=='full':
        list1 = [usr,tau,hsb,hlb,hlwebb,tsr,qsr,zot,zoq,Cd,Ch,Ce,L,zet]
        list2 = [dter,dqer,tkt,RF,Cdn_10,Chn_10,Cen_10]
        out = tuple(list1 + list2)
        A = np.column_stack(out)
    elif out=='min':
        list1 = [usr,tau]
        out = tuple(list1)
        A = np.column_stack(out)
    return A


def c36(u, zu , t, zt, rh, zq, P, ts, sw_dn, lw_dn, lat, lon,jd, zi,rain, Ss,
         cp=None, sigH=None, zrf_u=10.0, zrf_t=10.0, zrf_q=10.0, out='min'):   
    #**************************************************************************
    # VERSION INFO:
        
    # Vectorized version of COARE 3.6 code (Fairall et al, 2003) with
    # modification based on the CLIMODE, MBL and CBLAST experiments
    # (Edson et al., 2012). The cool skin and surface wave options are included.
    # A separate warm layer function can be used to call this function.
        
    # This 3.6 version include parameterizations using wave height and wave
    # slope using cp and sigH.  If these are set to NaN, then the wind
    # speed dependent formulation is used.  The parameterizations are based
    # on fits to the Banner-Norison wave model and the Fairall-Edson flux
    # database.  This version also allows salinity as a input.
    # Open ocean example Ss=35; Great Lakes Ss=0;
        
    #**************************************************************************
    # COOL SKIN:
        
    # An important component of this code is whether the inputed ts
    # represents the ocean skin temperature or a subsurface temperature.
    # How this variable is treated is determined by the jcool parameter:
    #   set jcool=1 if ts is subsurface or bulk ocean temperature (default);
    #   set jcool=0 if ts is skin temperature or to not run cool skin model.
    # The code updates the cool-skin temperature depression dT_skin and
    # thickness dz_skin during iteration loop for consistency. The number of
    # iterations set to nits = 6.
        
    jcoolx = 1

    #**************************************************************************
    ### INPUTS:
        
    # Notes on input default values, missing values, vectors vs. single values:
    #   - the code assumes u,t,rh,ts,P,sw_dn,lw_dn,rain,Ss,cp,sigH are vectors;
    #   - sensor heights (zu,zt,zl) latitude lat, longitude lon, julian date jd,
    #       and PBL height zi may be constants;
    #   - air pressure P and radiation sw_dn, lw_dn may be vectors or constants.
    #   - input NaNs as vectors or single values to indicate no data.
    #   - assign a default value to P, lw_dn, sw_dn, lat, zi if unknown, single
    #       values of these inputs are okay.
    # Notes about signs and units:
    #   - radiation signs: positive warms the ocean
    #   - signs and units change throughout the program for ease of calculations.
    #   - the signs and units noted here are for the inputs.
        
        
    #  u = water-relative wind speed magnitude (m/s) at height zu (m)
    #             i.e. mean wind speed accounting for the ocean current vector.
    #             i.e. the magnitude of the difference between the wind vector
    #             (at height zu) and ocean surface current vector.
    #             If not available, use true wind speed to compute fluxes in
    #             earth-coordinates only which will be ignoring the stress
    #             contribution from the ocean current to all fluxes
    #  t = air temperature (degC) at height zt (m)
    #  rh = relative humidity (#) at height zq (m)
    #  P = sea level air pressure (mb)
    #  ts = seawater temperature (degC), see jcool below for cool skin
    #             calculation, and separate warm layer code for specifying
    #             sensor depth and whether warm layer is computed
    #  sw_dn = downward (positive) shortwave radiation (W/m^2)
    #  lw_dn = downward (positive) longwave radiation (W/m^2)
    #  lat = latitude defined positive to north
    #  lon = longitude defined positive to east, if using other version,
    #             adjust the eorw string input to albedo_vector function
    #  jd = year day or julian day, where day Jan 1 00:00 UTC = 0
    #  zi = PBL height (m) (default or typical value = 600m)
    #  rain = rain rate (mm/hr)
    #  Ss = sea surface salinity (PSU)
    #  cp = phase speed of dominant waves (m/s) computed from peak period
    #  sigH = significant wave height (m)
    #  zu, zt, zq heights of the observations (m)
    #  zrf_u, zrf_t, zrf_q  reference height for profile.  Use this to compare observations at different heights
        
    #**************************************************************************
    #### OUTPUTS: the user controls the output array A at the end of the code.
        
    # Note about signs and units:
    #   - radiation signs: positive warms the ocean
    #   - sensible, rain, and latent flux signs: positive cools the ocean
    #   - signs and units change throughout the program for ease of calculations.
    #   - the signs and units noted here are for the final outputs.
        
    #    usr = friction velocity that includes gustiness (m/s), u*
    #    tau = wind stress that includes gustiness (N/m^2)
    #    hsb = sensible heat flux (W/m^2) ... positive for Tair < Tskin
    #    hlb = latent heat flux (W/m^2) ... positive for qair < qs
    #    hbb = atmospheric buoyany flux (W/m^2)... positive when hlb and hsb heat the atmosphere
    #    hsbb = atmospheric buoyancy flux from sonic ... as above, computed with sonic anemometer T
    #    hlwebb = webb factor to be added to hl covariance and ID latent heat fluxes
    #    tsr = temperature scaling parameter (K), t*
    #    qsr = specific humidity scaling parameter (kg/kg), q*
    #    zo = momentum roughness length (m)
    #    zot = thermal roughness length (m)
    #    zoq = moisture roughness length (m)
    #    Cd = wind stress transfer (drag) coefficient at height zu (unitless)
    #    Ch = sensible heat transfer coefficient (Stanton number) at height zu (unitless)
    #    Ce = latent heat transfer coefficient (Dalton number) at height zu (unitless)
    #    L = Monin-Obukhov length scale (m)
    #    zeta = Monin-Obukhov stability parameter zu/L (dimensionless)
    #    dT_skin = cool-skin temperature depression (degC), pos value means skin is cooler than subskin
    #    dq_skin = cool-skin humidity depression (g/kg)
    #    dz_skin = cool-skin thickness (m)
    #    Urf = wind speed at reference height (user can select height at input)
    #    Trf = air temperature at reference height
    #    Qrf = air specific humidity at reference height
    #    RHrf = air relative humidity at reference height
    #    UrfN = neutral value of wind speed at reference height
    #    TrfN = neutral value of air temp at reference height
    #    qarfN = neutral value of air specific humidity at reference height
    #    lw_net = Net IR radiation computed by COARE (W/m2)... positive heating ocean
    #    sw_net = Net solar radiation computed by COARE (W/m2)... positive heating ocean
    #    Le = latent heat of vaporization (J/K)
    #    rhoa = density of air at input parameter height zt, typically same as zq (kg/m3)
    #    UN = neutral value of wind speed at zu (m/s)
    #    U10 = wind speed adjusted to 10 m (m/s)
    #    UN10 = neutral value of wind speed at 10m (m/s)
    #    Cdn_10 = neutral value of drag coefficient at 10m (unitless)
    #    Chn_10 = neutral value of Stanton number at 10m (unitless)
    #    Cen_10 = neutral value of Dalton number at 10m (unitless)
    #    hrain = rain heat flux (W/m^2)... positive cooling ocean
    #    Qs = sea surface specific humidity, i.e. assuming saturation (g/kg)
    #    Evap = evaporation rate (mm/h)
    #    T10 = air temperature at 10m (deg C)
    #    Q10 = air specific humidity at 10m (g/kg)
    #    RH10 = air relative humidity at 10m (#)
    #    P10 = air pressure at 10m (mb)
    #    rhoa10 = air density at 10m (kg/m3)
    #    gust = gustiness velocity (m/s)
    #    wc_frac = whitecap fraction (ratio)
    #    Edis = energy dissipated by wave breaking (W/m^2)
        
    #**************************************************************************
    #### ADDITONAL CALCULATIONS:
        
    #   using COARE output, one can easily calculate the following using the
    #   sign conventions and names herein:
        
    #     ### Skin sea surface temperature or interface temperature; neglect
    #     ### dT_warm_to_skin if warm layer code is not used as the driver
    #     #### program for this program
    #           Tskin = ts + dT_warm_to_skin - dT_skin;
        
    #     ### Upwelling radiative fluxes: positive heating the ocean
    #           lw_up = lw_net - lw_dn;
    #           sw_up = sw_net - sw_dn;
        
    #     ### Net heat flux: positive heating ocean
    #     ### note that hs, hl, hrain are defined when positive cooling
    #     ### ocean by COARE, so their signs are flipped here:
    #           hnet = sw_net + lw_net - hs - hl - hrain;
        
    #**************************************************************************
    #### REFERENCES:
        
        #  Fairall, C. W., E. F. Bradley, J. S. Godfrey, G. A. Wick, J. B. Edson,
    #  and G. S. Young, 1996a: Cool-skin and warm-layer effects on sea surface
    #  temperature. J. Geophys. Res., 101, 1295?1308.
        
        #  Fairall, C. W., E. F. Bradley, D. P. Rogers, J. B. Edson, and G. S. Young,
    #  1996b: Bulk parameterization of air-sea fluxes for Tropical Ocean- Global
    #  Atmosphere Coupled- Ocean Atmosphere Response Experiment. J. Geophys. Res.,
    #  101, 3747?3764.
        
        #  Fairall, C. W., A. B. White, J. B. Edson, and J. E. Hare, 1997: Integrated
    #  shipboard measurements of the marine boundary layer. Journal of Atmospheric
    #  and Oceanic Technology, 14, 338?359
        
        #  Fairall, C.W., E.F. Bradley, J.E. Hare, A.A. Grachev, and J.B. Edson (2003),
    #  Bulk parameterization of air sea fluxes: updates and verification for the
    #  COARE algorithm, J. Climate, 16, 571-590.
        
        #  Edson, J.B., J. V. S. Raju, R.A. Weller, S. Bigorre, A. Plueddemann, C.W.
    #  Fairall, S. Miller, L. Mahrt, Dean Vickers, and Hans Hersbach, 2013: On
    #  the Exchange of momentum over the open ocean. J. Phys. Oceanogr., 43,
    #  15891610. doi: http://dx.doi.org/10.1175/JPO-D-12-0173.1
        
    #**************************************************************************
    # CODE HISTORY:
        
    # 1. 12/14/05 - created based on scalar version coare26sn.m with input
    #    on vectorization from C. Moffat.
    # 2. 12/21/05 - sign error in psiu_26_3p6 corrected, and code added to use variable
    #    values from the first pass through the iteration loop for the stable case
    #    with very thin M-O length relative to zu (zetau>50) (as is done in the
    #    scalar coare26sn and COARE3 codes).
    # 3. 7/26/11 - S = dT was corrected to read S = ut.
    # 4. 7/28/11 - modification to roughness length parameterizations based
    #    on the CLIMODE, MBL, Gasex and CBLAST experiments are incorporated
    # 5. 9/20/2017 - New wave parameterization added based on fits to wave model
    # 6. 9/2020 - tested and updated to give consistent readme info and units,
    #    and so that no external functions are required. They are all included at
    #    end of this program now. Changed names for a few things... including skin
    #    dter -> dT_skin; dt -> dT; dqer -> dq_skin; tkt -> dz_skin
    #    and others to avoid ambiguity:
    #    Rnl -> lw_net; Rns -> sw_net; Rl -> lw_dn; Rs -> sw_dn;
    #    SST -> Tskin; Also corrected heights at which q and P are
    #    computed to be more accurate, changed units of qstar to kg/kg, removed
    #    extra 1000 on neutral 10 m transfer coefficients;
    # 7. 10/2021 - implemented zenith angle dependent sw_up and sw_net;
    #    changed buoyancy flux calculation to follow Stull
    #    textbook version of tv* and tv_sonic*; reformatted preamble of program for
    #    consistent formatting and to reduce redundancy; resolved issues of
    #    nomenclature around T adjusted to heights vs theta potential
    #    temperature when computing dT for the purpose of sensible heat flux.
    #-----------------------------------------------------------------------

    #***********  prep input data *********************************************
        
    ### Make sure INPUTS are consistent in size. 
    # Best to avoid NaNs as inputs as well. Will prevent weird results

    # be sure array inputs are ndarray floats for single value function
    # if inputs are already ndarray float this does nothing
    # otherwise copies are created in the local namespace
    # .flatten() return a 1D version in case single value input is already an array (array([[]]) vs array([]))
    if u.size ==1 and t.size ==1: 
        u = np.copy(np.asarray([u], dtype=float)).flatten()
        zu = np.copy(np.asarray([zu], dtype=float)).flatten()
        t = np.copy(np.asarray([t], dtype=float)).flatten()
        zt = np.copy(np.asarray([zt], dtype=float)).flatten()
        rh = np.copy(np.asarray([rh], dtype=float)).flatten()
        zq = np.copy(np.asarray([zq], dtype=float)).flatten()
        P = np.copy(np.asarray([P], dtype=float)).flatten()
        ts = np.copy(np.asarray([ts], dtype=float)).flatten()
        sw_dn = np.copy(np.asarray([sw_dn], dtype=float)).flatten()
        lw_dn = np.copy(np.asarray([lw_dn], dtype=float)).flatten()
        lat = np.copy(np.asarray([lat], dtype=float)).flatten()
        lon = np.copy(np.asarray([lon], dtype=float)).flatten()
        jd = np.copy(np.asarray([jd], dtype=float)).flatten()
        zi = np.copy(np.asarray([zi], dtype=float)).flatten()
        rain = np.copy(np.asarray([rain], dtype=float)).flatten()
        Ss = np.copy(np.asarray([Ss], dtype=float)).flatten()
        zrf_u = np.copy(np.asarray([zrf_u], dtype=float)).flatten()
        zrf_t = np.copy(np.asarray([zrf_t], dtype=float)).flatten()
        zrf_q = np.copy(np.asarray([zrf_q], dtype=float)).flatten()
    
    N = np.size(u)
    jcool = jcoolx * np.ones(N)
    
    if cp is not None and cp.size==1:
        cp = np.copy(np.asarray([cp], dtype=float)).flatten()
    elif cp is None:
        cp = np.nan * np.ones(N)

    if sigH is not None and sigH.size==1:
        sigH = np.copy(np.asarray([sigH], dtype=float)).flatten()
    elif sigH is None:
        sigH = np.nan * np.ones(N)
     
    # Option to set local variables to default values if input is NaN... can do
    # single value or fill each individual. Warning... this will fill arrays
    # with the dummy values and produce results where no input data are valid
    # ii=find(isnan(P)); P(ii)=1013;    # pressure
    # ii=find(isnan(sw_dn)); sw_dn(ii)=200;   # incident shortwave radiation
    # ii=find(isnan(lat)); lat(ii)=45;  # latitude
    # ii=find(isnan(lw_dn)); lw_dn(ii)=400-1.6*abs(lat(ii)); # incident longwave radiation
    # ii=find(isnan(zi)); zi(ii)=600;   # PBL height
    # ii=find(isnan(Ss)); Ss(ii)=35;    # Salinity
        
    # find missing input data
    # iip = np.where(np.isnan(P))
    # iirs = np.where(np.isnan(sw_dn))
    # iilat = np.where(np.isnan(lat))
    # iirl = np.where(np.isnan(lw_dn))
    # iizi = np.where(np.isnan(zi))
    # iiSs = np.where(np.isnan(Ss))
    # Input variable u is assumed to be wind speed corrected for surface current
    # (magnitude of difference between wind and surface current vectors). To
    # follow orginal Fairall code, set surface current speed us=0. If us surface
    # current data are available, construct u prior to using this code and
    # input us = 0*u here;
    us = 0 * u
    # convert rh to specific humidity after accounting for salt effect on freezing
    # point of water
    Tf = - 0.0575 * Ss + 0.00171052 * Ss ** 1.5 - np.multiply(0.0002154996 * Ss,Ss)
    Qs = qsat26sea(ts,P,Ss,Tf) / 1000
    P_tq = P - (0.125 * zt)
    Q,Pv = qsat26air(t,P_tq,rh)
    
    # Assumes rh relative to ice T<0
    # Pv is the partial pressure due to wate vapor in mb
    Q = Q / 1000

    ice = np.zeros(N)
    iice = np.array(np.where(ts < Tf))
    ice[iice] = 1
    jcool[iice] = 0
    zos = 0.0005
    #***********  set constants ***********************************************
    zref = 10
    Beta = 1.2
    von = 0.4
    fdg = 1.0
    T2K = 273.16
    grav = grv(lat)
    #***********  air constants ***********************************************
    Rgas = 287.1
    Le = (2.501 - 0.00237 * ts) * 1000000.0
    cpa = 1004.67
    cpv = cpa * (1 + 0.84 * Q)
    rhoa = P_tq * 100.0 / (np.multiply(Rgas * (t + T2K),(1 + 0.61 * Q)))
    # Pv is the partial pressure due to wate vapor in mb
    rhodry = (P_tq - Pv) * 100.0 / (Rgas * (t + T2K))
    visa = 1.326e-05 * (1 + np.multiply(0.006542,t) + 8.301e-06 * t ** 2 - 4.84e-09 * t ** 3)
    lapse = grav / cpa
    
    #***********  cool skin constants  ***************************************
    ### includes salinity dependent thermal expansion coeff for water
    tsw = ts
    ii = np.array(np.where(ts < Tf))
    if np.size(ii) != 0:
        tsw[ii] = Tf[ii]
    Al35 = 2.1e-05 * (tsw + 3.2) ** 0.79
    Al0_i=(tsw - 1) ** 0.82
    Al0 = (2.2 * Al0_i.real - 5) * 1e-05
    Al = Al0 + np.multiply((Al35 - Al0),Ss) / 35
    ###################
    bets = 0.00075
    be = bets * Ss
    ####  see "Computing the seater expansion coefficients directly from the
    ####  1980 equation of state".  J. Lillibridge, J.Atmos.Oceanic.Tech, 1980.
    cpw = 4000
    rhow = 1022
    visw = 1e-06
    tcw = 0.6
    bigc = 16 * grav * cpw * (rhow * visw) ** 3.0 / (tcw ** 2 * rhoa ** 2)
    wetc = np.multiply(0.622 * Le,Qs) / (Rgas * (ts + T2K) ** 2)
    #***********  net solar and IR radiation fluxes ***************************
    ### net solar flux, aka sw, aka shortwave
    
    # *** for time-varying, i.e. zenith angle varying albedo using Payne 1972:
    # insert 'E' for input to albedo function if longitude is defined positive
    # to E (normal), in this case lon sign will be flipped for the calculation.
    # Otherwise specify 'W' and the sign will not be changed in the function.
    # Check: albedo should usually peak at sunrise not at sunset, though it may
    # vary based on sw_dn.
    alb,T_sw,solarmax_sw,psi_sw = albedo_vector(sw_dn,jd,lon,lat,eorw='E')
    sw_net = np.multiply((1 - alb),sw_dn)
    
    # *** for constant albedo:
    # sw_net = 0.945.*sw_dn; # constant albedo correction, positive heating ocean
    
    ### net longwave aka IR aka infrared
    # initial value here is positive for cooling ocean in the calculations
    # below. However it is returned at end of program as -lw_net in final output so
    # that it is positive heating ocean like the other input radiation values.
    lw_net = 0.97 * (5.67e-08 * (ts - 0.3 * jcool + T2K) ** 4 - lw_dn)
    #***********  begin bulk loop *********************************************
    
    #***********  first guess *************************************************
    
    # wind speed minus current speed
    du = u - us
    # air sea temperature difference for the purpose of sensible heat flux
    dT = ts - t - np.multiply(lapse,zt)
    # air-sea T diff must account for lapse rate between surface and instrument height
    # t is air temperature in C, ts is surface water temperature in C. dT is
    # an approximation that is equivalent to  dtheta where theta is the
    # potential temperature, and the pressure at sea level and instrument level
    # are used. They are equivalent (max difference = 0.0022 K). This way
    # elimniates the need to involve the pressures at different heights.
    # Using or assuming dry adiabatic lapse rate between the two heights
    # doesn't matter because if real pressures are used the result is the
    # unchanged. The dT need not include conversion to K either. Here's an example:
    # grav = grv(lat);
    # lapse=grav/cpa;
    # P_at_tq_height=(psealevel - (0.125*zt)); # P at tq measurement height (mb)
    # note psealevel is adjusted using same expression from pa height
    # Ta is originally in C and C2K = 273.15 to convert from C to K
    # theta = (b10.Ta+C2K).*(1000./P_tq).^(Rgas/cpa);
    # TadjK = (b10.Ta+C2K) + lapse*zt;
    # Tadj = b10.Ta + lapse*zt;
    # theta_sfc = (b10.Tskin+C2K).*(1000./b10.psealevel).^(Rgas/cpa);
    # TadjK_sfc = b10.Tskin+C2K;
    # Tadj_sfc = b10.Tskin;
        
    ### the adj versions are only 0.0022 K smaller than theta versions)
    # dtheta = theta_sfc - theta;
    # dTadjK = TadjK_sfc - TadjK;
    # dTadj = Tadj_sfc - Tadj; # so dT = Tskin - (Ta + lapse*zt) = Tskin - Ta - lapse*zt
        
    # put things into different units and expressions for more calculations,
    # including first guesses that get redone later
    dq = Qs - Q
    ta = t + T2K
    tv = np.multiply(ta,(1 + 0.61 * Q))
    gust = 0.5
    dT_skin = 0.3
    ut = np.sqrt(du ** 2 + gust ** 2)
    u10 = np.multiply(ut,np.log(10 / 0.0001)) / np.log(zu / 0.0001)
    usr = 0.035 * u10
    zo10 = 0.011 * usr ** 2.0 / grav + 0.11 * visa / usr
    Cd10 = (von / np.log(10.0 / zo10)) ** 2
    Ch10 = 0.00115
    Ct10 = Ch10 / np.sqrt(Cd10)
    zot10 = 10.0 / np.exp(von / Ct10)
    Cd = (von / np.log(zu / zo10)) ** 2
    Ct = von / np.log(zt / zot10)
    CC = von * Ct / Cd
    Ribcu = - zu / zi / 0.004 / Beta ** 3
    Ribu = np.multiply(np.multiply(- grav,zu) / ta,((dT - np.multiply(dT_skin,jcool)) + np.multiply(0.61 * ta,dq))) / ut ** 2
    zetau = np.multiply(np.multiply(CC,Ribu),(1 + 27 / 9 * Ribu / CC))
    k50 = np.array(np.where(zetau > 50))
    
    k = np.array(np.where(Ribu < 0))
    if np.size(Ribcu) == 1:
        zetau[k] = np.multiply(CC[k],Ribu[k]) / (1 + Ribu[k] / Ribcu)
        del k
    else:
        zetau[k] = np.multiply(CC[k],Ribu[k]) / (1 + Ribu[k] / Ribcu[k])
        del k
    
    L10 = zu / zetau
    gf = ut / du
    usr = np.multiply(ut,von) / (np.log(zu / zo10) - psiu_40_3p6(zu / L10))
    tsr = np.multiply(- (dT - np.multiply(dT_skin,jcool)),von) * fdg / (np.log(zt / zot10) - psit_26_3p6(zt / L10))
    qsr = - (dq - np.multiply(np.multiply(wetc,dT_skin),jcool)) * von * fdg / (np.log(zq / zot10) - psit_26_3p6(zq / L10))
    dz_skin = 0.001 * np.ones(N)
    #  The following gives the new formulation for the Charnock variable
    #############   COARE 3.5 wind speed dependent charnock
    charnC = 0.011 * np.ones(N)
    umax = 19
    a1 = 0.0017
    a2 = - 0.005
    # charnC = a1 * u10 + a2
    charnC=np.copy(np.asarray(a1 * u10 + a2, dtype=float))
    k = np.array(np.where(u10 > umax))
    if k.size!=0:
        charnC[k] = a1 * umax + a2
    #########   if wave age is given but not wave height, use parameterized
    #########   wave height based on wind speed
    hsig = np.multiply((0.02 * (cp / u10) ** 1.1 - 0.0025),u10 ** 2)
    hsig = np.maximum(hsig,0.25)
    ii = np.array(np.where(np.logical_and(np.logical_not(np.isnan(cp)) ,np.isnan(sigH))))
    if ii.size!=0:
        sigH[ii] = hsig[ii]
    Ad = 0.2
    Bd = 2.2
    zoS = np.multiply(np.multiply(sigH,Ad),(usr / cp) ** Bd)
    charnS = np.multiply(zoS,grav) / usr / usr
    nits = 10

    # creates a deep copy of charnC - if shallow copy (= only) charnC may change too below!
    charn = np.copy(charnC)  
    ii = np.array(np.where(np.logical_not(np.isnan(cp))))
    charn[ii] = charnS[ii]
    #**************  bulk loop ************************************************
    
    for i in np.arange(1,nits+1).reshape(-1):
        zeta = np.multiply(np.multiply(np.multiply(von,grav),zu) / ta,(tsr + np.multiply(0.61 * ta,qsr))) / (usr ** 2)
        L = zu / zeta
        zo = np.multiply(charn,usr ** 2.0) / grav + 0.11 * visa / usr
        zo[iice] = zos
        rr = np.multiply(zo,usr) / visa
        rt = np.zeros(u.size)
        rq = np.zeros(u.size)
        # This thermal roughness length Stanton number is close to COARE 3.0 value
        zoq = np.minimum(0.00016,5.8e-05 / rr ** 0.72)
        # Andreas 1987 for snow/ice
        ik = np.array(np.where(rr[iice] <= 0.135))
        rt[iice[ik]] = rr[iice[ik]] * np.exp(1.25)
        rq[iice[ik]] = rr[iice[ik]] * np.exp(1.61)
        ik = np.array(np.where(rr[iice] > np.logical_and(0.135,rr[iice]) <= 2.5))
        rt[iice[ik]] = np.multiply(rr[iice[ik]],np.exp(0.149 - 0.55 * np.log(rr[iice[ik]])))
        rq[iice[ik]] = np.multiply(rr[iice[ik]],np.exp(0.351 - 0.628 * np.log(rr[iice[ik]])))
        ik = np.array(np.where(rr[iice] > np.logical_and(2.5,rr[iice]) <= 1000))
        rt[iice[ik]] = np.multiply(rr[iice[ik]],np.exp(0.317 - 0.565 * np.log(rr[iice[ik]]) - np.multiply(0.183 * np.log(rr[iice[ik]]),np.log(rr[iice[ik]]))))
        rq[iice[ik]] = np.multiply(rr[iice[ik]],np.exp(0.396 - 0.512 * np.log(rr[iice[ik]]) - np.multiply(0.18 * np.log(rr[iice[ik]]),np.log(rr[iice[ik]]))))
        # Dalton number is close to COARE 3.0 value
        zot = zoq
        cdhf = von / (np.log(zu / zo) - psiu_26_3p6(zu / L))
        cqhf = np.multiply(von,fdg) / (np.log(zq / zoq) - psit_26_3p6(zq / L))
        cthf = np.multiply(von,fdg) / (np.log(zt / zot) - psit_26_3p6(zt / L))
        usr = np.multiply(ut,cdhf)
        qsr = np.multiply(- (dq - np.multiply(np.multiply(wetc,dT_skin),jcool)),cqhf)
        tsr = np.multiply(- (dT - np.multiply(dT_skin,jcool)),cthf)
        # original COARE version buoyancy flux
        tvsr1 = tsr + np.multiply(0.61 * ta,qsr)
        tssr1 = tsr + np.multiply(0.51 * ta,qsr)
        # new COARE version buoyancy flux from Stull (1988) page 146
        # tsr here uses dT with the lapse rate adjustment (see code above). The
        # Q and ta values should be at measurement height, not adjusted heights
        tvsr = np.multiply(tsr,(1 + np.multiply(0.61,Q))) + np.multiply(0.61 * ta,qsr)
        tssr = np.multiply(tsr,(1 + np.multiply(0.51,Q))) + np.multiply(0.51 * ta,qsr)
        Bf = np.multiply(np.multiply(- grav / ta,usr),tvsr)
        gust = 0.2 * np.ones(N)
        k = np.array(np.where(Bf > 0))
        ### gustiness in this way is from the original code. Notes:
        # we measured the actual gustiness by measuring the variance of the
        # wind speed and empirically derived the the scaling. It's empirical
        # but it seems appropriate... the longer the time average then the larger
        # the gustiness factor should be, to account for the gustiness averaged
        # or smoothed out by the averaging. wstar is the convective velocity.
        # gustiness is beta times wstar. gustiness is different between mean of
        # velocity and square of the mean of the velocity vector components.
        # The actual wind (mean + fluctuations) is still the most relavent
        # for the flux. The models do u v w, and then compute vector avg to get
        # speed, so we've done the same thing. coare alg input is the magnitude
        # of the mean vector wind relative to water.
        if np.size(zi) == 1:
            gust[k] = Beta * (np.multiply(Bf[k],zi)) ** 0.333
            del k
        else:
            gust[k] = Beta * (np.multiply(Bf[k],zi[k])) ** 0.333
            del k
        ut = np.sqrt(du ** 2 + gust ** 2)
        gf = ut / du
        hsb = np.multiply(np.multiply(- rhoa * cpa,usr),tsr)
        hlb = np.multiply(np.multiply(np.multiply(- rhoa,Le),usr),qsr)
        qout = lw_net + hsb + hlb
        ### rain heat flux is not included in qout because we don't fully
        # understand the evolution or gradient of the cool skin layer in the
        # presence of rain, and the sea snake subsurface measurement input
        # value will capture some of the rain-cooled water already. TBD.
        ### solar absorption:
        # The absorption function below is from a Soloviev paper, appears as
        # Eq 17 Fairall et al. 1996 and updated/tested by Wick et al. 2005. The
        # coefficient was changed from 1.37 to 0.065 ~ about halved.
        # Most of the time this adjustment makes no difference. But then there
        # are times when the wind is weak, insolation is high, and it matters a
        # lot. Using the original 1.37 coefficient resulted in many unwarranted
        # warm-skins that didn't seem realistic. See Wick et al. 2005 for details.
        # That's the last time the cool-skin routine was updated. The
        # absorption is not from Paulson & Simpson because that was derived in a lab.
        # It absorbed too much and produced too many warm layers. It likely
        # approximated too much near-IR (longerwavelength solar) absorption
        # which probably doesn't make it to the ocean since it was probably absorbed
        # somewhere in the atmosphere first. The below expression could
        # likely use 2 exponentials if you had a shallow mixed layer...
        # but we find better results with 3 exponentials. That's the best so
        # far we've found that covers the possible depths.
        dels = np.multiply(sw_net,(0.065 + 11 * dz_skin - np.multiply(6.6e-05 / dz_skin,(1 - np.exp(- dz_skin / 0.0008)))))
        qcol = qout - dels
        # only needs stress, water temp, sum of sensible, latent, ir, solar,
        # and latent individually.
        alq = np.multiply(Al,qcol) + np.multiply(np.multiply(be,hlb),cpw) / Le
        xlamx = 6.0 * np.ones(N)
        #     the other is the salinity part caused by latent heat flux (evap) leaving behind salt.
        dz_skin = np.minimum(0.01,np.multiply(xlamx,visw) / (np.multiply(np.sqrt(rhoa / rhow),usr)))
        k = np.array(np.where(alq > 0))
        xlamx[k] = 6.0 / (1 + (np.multiply(bigc[k],alq[k]) / usr[k] ** 4) ** 0.75) ** 0.333
        dz_skin[k] = np.multiply(xlamx[k],visw) / (np.multiply(np.sqrt(rhoa[k] / rhow),usr[k]))
        del k
        dT_skin = np.multiply(qcol,dz_skin) / tcw
        dq_skin = np.multiply(wetc,dT_skin)
        lw_net = 0.97 * (5.67e-08 * (ts - np.multiply(dT_skin,jcool) + T2K) ** 4 - lw_dn)
        if i == 1:
            usr50 = usr[k50]
            tsr50 = tsr[k50]
            qsr50 = qsr[k50]
            L50 = L[k50]
            zeta50 = zeta[k50]
            dT_skin50 = dT_skin[k50]
            dq_skin50 = dq_skin[k50]
            tkt50 = dz_skin[k50]
        u10N = np.multiply(usr / von / gf,np.log(10.0 / zo))
        charnC = a1 * u10N + a2
        k = u10N > umax
        charnC[k] = a1 * umax + a2
        charn = charnC
        zoS = np.multiply(np.multiply(sigH,Ad),(usr / cp) ** Bd)
        charnS = np.multiply(zoS,grav) / usr / usr
        ii = np.array(np.where(np.logical_not(np.isnan(cp))))
        charn[ii] = charnS[ii]
    
    # end bulk loop
    
    # insert first iteration solution for case with zetau>50
    usr[k50] = usr50
    tsr[k50] = tsr50
    qsr[k50] = qsr50
    L[k50] = L50
    zeta[k50] = zeta50
    dT_skin[k50] = dT_skin50
    dq_skin[k50] = dq_skin50
    dz_skin[k50] = tkt50
    #****************  compute fluxes  ****************************************
    tau = np.multiply(np.multiply(rhoa,usr),usr) / gf
    
    hsb = np.multiply(np.multiply(np.multiply(- rhoa,cpa),usr),tsr)
    
    hlb = np.multiply(np.multiply(np.multiply(- rhoa,Le),usr),qsr)
    
    hbb = np.multiply(np.multiply(np.multiply(- rhoa,cpa),usr),tvsr)
    
    hbb1 = np.multiply(np.multiply(np.multiply(- rhoa,cpa),usr),tvsr1)
    
    hsbb = np.multiply(np.multiply(np.multiply(- rhoa,cpa),usr),tssr)
    
    hsbb1 = np.multiply(np.multiply(np.multiply(- rhoa,cpa),usr),tssr1)
    
    wbar = 1.61 * hlb / Le / (1 + 1.61 * Q) / rhoa + hsb / rhoa / cpa / ta
    
    hlwebb = np.multiply(np.multiply(np.multiply(rhoa,wbar),Q),Le)
    
    Evap = 1000 * hlb / Le / 1000 * 3600
    
    #*****  compute transfer coeffs relative to ut @ meas. ht  ****************
    Cd = tau / rhoa / ut / np.maximum(0.1,du)
    Ch = np.multiply(- usr,tsr) / ut / (dT - np.multiply(dT_skin,jcool))
    Ce = np.multiply(- usr,qsr) / (dq - np.multiply(dq_skin,jcool)) / ut
    #***##  compute 10-m neutral coeff relative to ut *************************
    Cdn_10 = von ** 2.0 / np.log(10.0 / zo) ** 2
    Chn_10 = von ** 2.0 * fdg / np.log(10.0 / zo) / np.log(10.0 / zot)
    Cen_10 = von ** 2.0 * fdg / np.log(10.0 / zo) / np.log(10.0 / zoq)
    #***##  compute 10-m neutral coeff relative to ut *************************
    
    # Find the stability functions for computing values at user defined
    # reference heights and 10 m
    psi = psiu_26_3p6(zu / L)
    psi10 = psiu_26_3p6(10.0 / L)
    psirf = psiu_26_3p6(zrf_u / L)
    psiT = psit_26_3p6(zt / L)
    psi10T = psit_26_3p6(10.0 / L)
    psirfT = psit_26_3p6(zrf_t / L)
    psirfQ = psit_26_3p6(zrf_q / L)
    gf = ut / du
    #*********************************************************
    #  Determine the wind speeds relative to ocean surface at different heights
    #  Note that usr is the friction velocity that includes
    #  gustiness usr = sqrt(Cd) S, which is equation (18) in
    #  Fairall et al. (1996)
    #*********************************************************
    S = ut
    U = du
    S10 = S + np.multiply(usr / von,(np.log(10.0 / zu) - psi10 + psi))
    U10 = S10 / gf
    # or U10 = U + usr./von./gf.*(log(10/zu)-psi10+psi);
    Urf = U + np.multiply(usr / von / gf,(np.log(zrf_u / zu) - psirf + psi))
    UN = U + np.multiply(psi,usr) / von / gf
    U10N = U10 + np.multiply(psi10,usr) / von / gf
    
    UrfN = Urf + np.multiply(psirf,usr) / von / gf
    UN2 = np.multiply(usr / von / gf,np.log(zu / zo))
    U10N2 = np.multiply(usr / von / gf,np.log(10.0 / zo))
    UrfN2 = np.multiply(usr / von / gf,np.log(zrf_u / zo))
    #******** rain heat flux *****************************
    dwat = 2.11e-05 * ((t + T2K) / T2K) ** 1.94
    dtmp = np.multiply((1.0 + 0.003309 * t - np.multiply(np.multiply(1.44e-06,t),t)),0.02411) / (np.multiply(rhoa,cpa))
    dqs_dt = np.multiply(Q,Le) / (np.multiply(Rgas,(t + T2K) ** 2))
    alfac = 1.0 / (1 + 0.622 * (np.multiply(np.multiply(dqs_dt,Le),dwat)) / (np.multiply(cpa,dtmp)))
    hrain = np.multiply(np.multiply(np.multiply(rain,alfac),cpw),((ts - t - np.multiply(dT_skin,jcool)) + np.multiply((Qs - Q - np.multiply(dq_skin,jcool)),Le) / cpa)) / 3600
    
    Tskin = ts - np.multiply(dT_skin,jcool)
    
    # P is sea level pressure, so use subtraction through hydrostatic equation
    # to get P10 and P at reference height
    P10 = P - (0.125 * 10)
    Prf = P - (0.125 * zref)
    T10 = t + np.multiply(tsr / von,(np.log(10.0 / zt) - psi10T + psiT)) + np.multiply(lapse,(zt - 10))
    Trf = t + np.multiply(tsr / von,(np.log(zrf_t / zt) - psirfT + psiT)) + np.multiply(lapse,(zt - zrf_t))
    TN = t + np.multiply(psiT,tsr) / von
    T10N = T10 + np.multiply(psi10T,tsr) / von
    TrfN = Trf + np.multiply(psirfT,tsr) / von
    # unused... these are here to make sure you gets the same answer whether
    # you used the thermal calculated roughness lengths or the values at the
    # measurement height. So at this point they are just illustrative and can
    # be removed or ignored if you want.
    TN2 = Tskin + np.multiply(tsr / von,np.log(zt / zot)) - np.multiply(lapse,zt)
    T10N2 = Tskin + np.multiply(tsr / von,np.log(10.0 / zot)) - np.multiply(lapse,10)
    TrfN2 = Tskin + np.multiply(tsr / von,np.log(zrf_t / zot)) - np.multiply(lapse,zrf_t)
    dq_skin = np.multiply(np.multiply(wetc,dT_skin),jcool)
    Qs = Qs - dq_skin
    dq_skin = dq_skin * 1000
    Qs = Qs * 1000
    Q = Q * 1000
    Q10 = Q + np.multiply(np.multiply(1000.0,qsr) / von,(np.log(10.0 / zq) - psi10T + psiT))
    Qrf = Q + np.multiply(np.multiply(1000.0,qsr) / von,(np.log(zrf_q / zq) - psirfQ + psiT))
    QN = Q + np.multiply(np.multiply(psiT,1000.0),qsr) / von / np.sqrt(gf)
    Q10N = Q10 + np.multiply(np.multiply(psi10T,1000.0),qsr) / von
    QrfN = Qrf + np.multiply(np.multiply(psirfQ,1000.0),qsr) / von
    # unused... these are here to make sure you gets the same answer whether
    # you used the thermal calculated roughness lengths or the values at the
    # measurement height. So at this point they are just illustrative and can
    # be removed or ignored if you want.
    QN2 = Qs + np.multiply(np.multiply(1000.0,qsr) / von,np.log(zq / zoq))
    Q10N2 = Qs + np.multiply(np.multiply(1000.0,qsr) / von,np.log(10.0 / zoq))
    QrfN2 = Qs + np.multiply(np.multiply(1000.0,qsr) / von,np.log(zrf_q / zoq))
    RHrf = rhcalc_3p6(Trf,Prf,Qrf / 1000,Tf)
    RH10 = rhcalc_3p6(T10,P10,Q10 / 1000,Tf)
    # recompute rhoa10 with 10-m values of everything else.
    rhoa10 = P10 * 100.0 / (np.multiply(Rgas * (T10 + T2K),(1 + 0.61 * (Q10 / 1000))))
    ############  Other wave breaking statistics from Banner-Morison wave model
    wc_frac = 0.00073 * (U10N - 2) ** 1.43
    wc_frac[U10 < 2.1] = 1e-05
    
    kk = np.array(np.where(np.isfinite(cp) == 1))
    wc_frac[kk] = 0.0016 * U10N[kk] ** 1.1 / np.sqrt(cp[kk] / U10N[kk])
    
    Edis = np.multiply(np.multiply(0.095 * rhoa,U10N),usr ** 2)
    wc_frac[iice] = 0
    Edis[iice] = 0
    #****************  output  ****************************************************
    # only return values if jcool = 1; if cool skin model was intended to be run
    dT_skinx = np.multiply(dT_skin,jcool)
    dq_skinx = np.multiply(dq_skin,jcool)
    # get rid of filled values where nans are present in input data
    bad_input = np.array(np.where(np.isnan(u) == 1))
    gust[bad_input] = np.nan
    dz_skin[bad_input] = np.nan
    zot[bad_input] = np.nan
    zoq[bad_input] = np.nan
    # flip lw_net sign for standard radiation sign convention: positive heating ocean
    lw_net = - lw_net
    # this sign flip means lw_net, net long wave flux, is equivalent to:
    # lw_net = 0.97*(lw_dn_best - 5.67e-8*(Tskin+C2K).^4);
    
    # adjust A output as desired:

    if out=='lim':
        out = np.array([usr,tau])
    if out=='full':
        out = np.array([usr,tau,hsb,hlb,hbb,hsbb,hlwebb,tsr,qsr,zo,zot,zoq,Cd,Ch,Ce,L,zeta,
                        dT_skinx,dq_skinx,dz_skin,Urf,Trf,Qrf,RHrf,UrfN,TrfN,QrfN,lw_net,sw_net,
                        Le,rhoa,UN,U10,U10N,Cdn_10,Chn_10,Cen_10,hrain,Qs,Evap,T10,T10N,Q10,Q10N,
                        RH10,P10,rhoa10,gust,wc_frac,Edis])
    
    A = np.column_stack(out)
    return A
    

def c36warm(Jd, U, Zu, Tair, Zt, RH, Zq, P, Tsea, SW_dn, LW_dn, Lat, Lon,
            Zi, Rainrate, Ts_depth, Ss, cp=None, sigH=None, zrf_u = 10.0, 
            zrf_t = 10.0, zrf_q = 10.0, out='min'): 
    #***********   input data **************
    #       Jd = day-of-year or julian day
    #	    U = wind speed magnitude (m/s) corrected for currents, i.e. relative to water at height zu
    #	    Zu = height (m) of wind measurement
    #	    Tair = air temp (degC) at height zt
    #	    Zt = height (m) of air temperature measurement
    #	    RH = relative humidity (#) at height zq
    #	    Zq = height (m) of air humidity measurement
    #	    P = air pressure at sea level (mb)
    #       Tsea = surface sea temp (degC) at ts_depth
    #	    SW_dn = downward solar flux (w/m^2) defined positive down
    #	    LW_dn = downward IR flux (w/m^2) defined positive down
    #	    Lat = latitude (deg N=+)
    #	    Lon = longitude (deg E=+) # If using other version, see
    #               usage of lon in albedo_vector function and adjust 'E' or
    #               'W' string input
    #       Zi = inversion height (m)
    #       Rainrate = rain rate (mm/hr)
    #       Ts_depth = depth (m) of water temperature measurement, positive for below
    #                   surface
    #       Ss = sea surface salinity (PSU)
    #       cp = phase speed of dominant waves (m/s)
    #       sigH = significant wave height (m)
    #       zu, zt, zq = heights of the observations (m)
    #       zrf_u, zrf_t, zrf_q = reference height for profile.
    #           Use this to compare observations at different heights
        
        
    #********** output data  ***************
    # Outputs
    # Adds onto output from coare36vn_zrf_et
    # .... see that function for updated output. It can change. This function adds 4 variables onto it:
    # previously named dt_wrm, tk_pwp, dsea, du_wrm... renamed to the following
    # for clarity:
    #         dT_warm = dT from base of warm layer to skin, i.e. warming across entire warm layer depth (deg C)
    #         dz_warm = warm layer thickness (m)
    #         dT_warm_to_skin = dT from measurement depth to skin due to warm layer,
    #                       such that Tskin = tsea + dT_warm_to_skin - dT_skin
    #         du_warm = total current accumulation in warm layer (m/s ?...  unsure of units but likely m/s)
        
    #********** history ********************
    # updated 09/2020 for consistency with units, readme info, and coare 3.6 main function
    #    Changed names for a few things...
    #    dt_wrm -> dT_warm; tk_pwp -> dz_warm; dsea -> dT_warm_to_skin;
    #    skin: dter -> dT_skin
    #    and fluxes to avoid ambiguity: RF -> hrain
    #    Rnl -> lw_net; Rns -> sw_net; Rl -> lw_dn; Rs -> sw_dn;
    #  updated 05/2022 - fix a glitch in code when using wave input - LB
    #   added     cpi=cp[ibg];   # air density
    #             sigHi=sigH[ibg];   # air density
    # and edited line 258 to use cpi and sigHi
    # Bx=coare36vn_zrf_et(u,zu,t,zt,rh,zq,p,ts,sw_dn,lw_dn,lat,lon,jd,zi,rain,ss,cpi,sigHi,zrf_u,zrf_t,zrf_q);
        
    #********** Set cool skin options ******************
    jcool = 1
    icount = 1
    #********** Set wave ******************
    ### ... not sure if this is necessary ...
    # if no wave info is provided, fill array with nan.
    # if length(cp) == 1 && isnan(cp) == 1
    #     cp = jd*nan;
    # end
    # if length(sigH) == 1 && isnan(sigH) == 1
    #     sigH = jd*nan;
    # end
    
    # be sure array inputs are ndarray floats for single value function
    # if inputs are already ndarray float this does nothing
    # otherwise copies are created in the local namespace
    if U.size ==1 and Tair.size ==1: 
        U = np.copy(np.asarray([U], dtype=float))
        Zu = np.copy(np.asarray([Zu], dtype=float))
        Tair = np.copy(np.asarray([Tair], dtype=float))
        Zt = np.copy(np.asarray([Zt], dtype=float))
        RH = np.copy(np.asarray([RH], dtype=float))
        Zq = np.copy(np.asarray([Zq], dtype=float))
        P = np.copy(np.asarray([P], dtype=float))
        Tsea = np.copy(np.asarray([Tsea], dtype=float))
        SW_dn = np.copy(np.asarray([SW_dn], dtype=float))
        LW_dn = np.copy(np.asarray([LW_dn], dtype=float))
        Lat = np.copy(np.asarray([Lat], dtype=float))
        Lon = np.copy(np.asarray([Lon], dtype=float))
        Jd = np.copy(np.asarray([Jd], dtype=float))
        Zi = np.copy(np.asarray([Zi], dtype=float))
        Rainrate = np.copy(np.asarray([Rainrate], dtype=float))
        Ss = np.copy(np.asarray([Ss], dtype=float))
        zrf_u = np.copy(np.asarray([zrf_u], dtype=float))
        zrf_t = np.copy(np.asarray([zrf_t], dtype=float))
        zrf_q = np.copy(np.asarray([zrf_q], dtype=float))
        Ts_depth = np.copy(np.asarray([Ts_depth], dtype=float))

    N = np.size(U)
 
    if cp is not None and cp.size==1:
        cp = np.copy(np.asarray([cp], dtype=float))
    elif cp is None:
        cp = np.nan * np.ones(N)
    
    if sigH is not None and sigH.size==1:
        sigH = np.copy(np.asarray([sigH], dtype=float))
    elif sigH is None:
        sigH = np.nan * np.ones(N)

    #   *********************  housekeep variables  ********
    # Call coare36vn to get initial flux values
    Bx = c36(U[0],Zu[0],Tair[0],Zt[0],RH[0],Zq[0],P[0],Tsea[0],SW_dn[0],LW_dn[0],Lat[0],Lon[0],Jd[0],Zi[0],Rainrate[0],Ss[0],cp[0],sigH[0],zrf_u,zrf_t,zrf_q)

    ### check these indices for you latest version of coare!
    tau_old = Bx[0,1]
    hs_old = Bx[0,2]
    hl_old = Bx[0,3]
    dT_skin_old = Bx[0,17]
    hrain_old = Bx[0,37]
    
    qcol_ac = 0.0
    tau_ac = 0.0
    dT_warm = 0.0
    du_warm = 0.0
    max_pwp = 19.0
    dz_warm = max_pwp
    dT_warm_to_skin = 0.0
    q_pwp = 0.0
    fxp = 0.5
    
    rich = 0.65
    
    jtime = 0
    jamset = 0
    jump = 1
    
    #*******************  set constants  ****************
    T2K = 273.16
    Rgas = 287.1
    cpa = 1004.67
    cpw = 4000.0
    rhow = 1022.0
    visw = 1e-06
    
    be = 0.026
    tcw = 0.6
    #******************  setup read data loop  ****************
    P_tq = P - (0.125 * Zt)
    Press,tseak,tairk,Qsatsea,Qsat,Qair,Rhoair,Rhodry = scalarv(P,Tsea,Tair,RH,Zt)

    nx = np.size(Jd)
    
    # this is an empty array for saving warm layer code output values. Will be
    # added to coare output at the end.
    warm_output = np.full([nx,4],np.nan)
    for ibg in np.arange(0,nx).reshape(-1):  #np.arange(0,53).reshape(-1):
        jd = Jd[ibg]
        p = P[ibg]
        u = U[ibg]
        tsea = Tsea[ibg]
        t = Tair[ibg]
        ss = Ss[ibg]
        qs = Qsatsea[ibg]
        q = Qair[ibg]
        rh = RH[ibg]
        sw_dn = SW_dn[ibg]
        lw_dn = LW_dn[ibg]
        rain = Rainrate[ibg]
        grav = grv(Lat[ibg])
        lat = Lat[ibg]
        lon = Lon[ibg]
        rhoa = Rhoair[ibg]
        cpi = cp[ibg]
        sigHi = sigH[ibg]
        zi=Zi[ibg]
        zu=Zu[ibg]
        zt=Zt[ibg]
        zq=Zq[ibg]
        ts_depth=Ts_depth[ibg]
        
        #*****  variables for warm layer  ***
        ### for constant albedo
        # sw_net=.945*sw_dn;     #Net Solar: positive warming ocean, constant albedo
        ### for albedo that is time-varying, i.e. zenith angle varying
        # insert 'E' for input to albedo function if longitude is defined positive
        # to E, so that lon can be flipped. The function ideally works with
        # longitude positive to the west. Check: albedo should peak at sunrise not
        # sunset.
        alb,T_sw,solarmax_sw,psi_sw = albedo_vector(sw_dn,jd,lon,lat,'E')
        sw_net = np.multiply((1 - alb[0]),sw_dn)
        lw_net = 0.97 * (5.67e-08 * (tsea - dT_skin_old * jcool + T2K) ** 4 - lw_dn)
        cpv = cpa * (1 + 0.84 * q / 1000)
        visa = 1.326e-05 * (1 + 0.006542 * t + 8.301e-06 * t * t - 4.84e-09 * t * t * t)
        Al = 2.1e-05 * (tsea + 3.2) ** 0.79
        ctd1 = np.sqrt(2 * rich * cpw / (Al * grav * rhow))
        ctd2 = np.sqrt(2 * Al * grav / (rich * rhow)) / (cpw ** 1.5)
        
        #****  Compute apply warm layer  correction *************
        intime = jd - np.fix(jd)
        loc = (lon + 7.5) / 15
        chktime = loc + intime * 24
        if chktime > 24:
            chktime = chktime - 24
        newtime = (chktime - 24 * np.fix(chktime / 24)) * 3600
        if icount > 1:
            if newtime <= 21600 or jump == 0:
                jump = 0
                if newtime < jtime:
                    jamset = 0
                    fxp = 0.5
                    dz_warm = max_pwp
                    tau_ac = 0.0
                    qcol_ac = 0.0
                    dT_warm = 0.0
                    du_warm = 0.0
                else:
                    #************************************
                    #****   set warm layer constants  ***
                    #************************************
                    dtime = newtime - jtime
                    qr_out = lw_net + hs_old + hl_old + hrain_old
                    q_pwp = fxp * sw_net - qr_out
                    # qqrx[ibg] = hs_old
                    if q_pwp >= 50 or jamset == 1:
                        jamset = 1
                        tau_ac = tau_ac + np.maximum(0.002,tau_old) * dtime
                        if qcol_ac + q_pwp * dtime > 0:
                            #******************************************
                            # Compute the absorption profile
                            #******************************************
                            for i in np.arange(1,5+1).reshape(-1):
                                #### The original version since Fairall et al. 1996:
                                fxp = 1 - (0.28 * 0.014 * (1 - np.exp(- dz_warm / 0.014)) 
                                           + 0.27 * 0.357 * (1 - np.exp(- dz_warm / 0.357)) 
                                           + 0.45 * 12.82 * (1 - np.exp(- dz_warm / 12.82))) / dz_warm
                                # the above integrated flux formulation is wrong for the warm layer,
                                # but it has been used in this scheme since 1996 without
                                # making bad predictions.
                                # Simon recognized that fxp should be the
                                # fraction absorbed in the layer of the form
                                # 1-sum(ai*exp(-tk_pwp/gammai)) with sum(ai)=1
                                # One can choose different exponential
                                # absorption parameters, but it has to be of the right form.
                                # Simon idealized coefficients from profiles of
                                # absorption in DYNAMO by C. Ohlmann.
                                # see /Users/sdeszoek/Data/cruises/DYNAMO_2011/solar_absorption/test_absorption_fcns.m
                                # Correct form of Soloviev 3-band absorption:
                                # --using original F96 absorption bands: # NOT TESTED!!
                                #  fxp=1-(0.32*exp(-tk_pwp/22.0) + 0.34*exp(-tk_pwp/1.2) + 0.34*exp(-tk_pwp/0.014)); 
                                # --using DYNAMO absorption bands (F, invgamma defined above):
                                #### NOT TESTED!! Correction of fxp from Simon ***
                                #fxp=1-sum(F.*(exp(-tk_pwp*invgamma)),2);
                                qjoule = (fxp * sw_net - qr_out) * dtime
                                if qcol_ac + qjoule > 0:
                                    dz_warm = np.minimum(max_pwp,ctd1 * tau_ac / np.sqrt(qcol_ac + qjoule))
                        else:
                            fxp = 0.75
                            dz_warm = max_pwp
                            qjoule = (fxp * sw_net - qr_out) * dtime
                        qcol_ac = qcol_ac + qjoule
                        #*******  compute dt_warm  ******
                        if qcol_ac > 0:
                            dT_warm = ctd2 * (qcol_ac) ** 1.5 / tau_ac
                            du_warm = 2 * tau_ac / (dz_warm * rhow)
                        else:
                            dT_warm = 0
                            du_warm = 0
                # Compute warm layer dT between input measurement and skin layer
                if dz_warm < ts_depth:
                    dT_warm_to_skin = dT_warm
                else:
                    dT_warm_to_skin = dT_warm * ts_depth / dz_warm
        jtime = newtime
        
        #************* output from routine  *****************************
        # Adjust tsea for warm layer above measurement. Even if tsea is at 5 cm for the sea snake,
        # there could be warming present between the interface and the subsurface
        # temperature. dT_warm_to_skin estimates that warming between the levels.
        ts = tsea + dT_warm_to_skin
        # Rerun COARE with the warm-layer corrected tsea temperature. COARE will
        # apply a cool skin to this, completing all calculations needed for Tskin and fluxes.
        # Using COARE ouput from this function, Tskin = Tsnake - dT_skin + dT_warm_to_skin
        # note: in prior COARE lingo/code: dT_warm_to_skin used to be dsea and dT_skin used to be dter
        Bx = c36(u,zu,t,zt,rh,zq,p,ts,sw_dn,lw_dn,lat,lon,jd,zi,rain,ss,cpi,sigHi,zrf_u,zrf_t,zrf_q)
        # save values from this time step to be used in next time step, this is how
        # the integral is computed
        tau_old = Bx[0,1]
        hs_old = Bx[0,2]
        hl_old = Bx[0,3]
        dT_skin_old = Bx[0,17]
        hrain_old = Bx[0,37]
        warm_output[ibg,0] = dT_warm
        warm_output[ibg,1] = dz_warm
        warm_output[ibg,2] = dT_warm_to_skin
        warm_output[ibg,3] = du_warm
        
        icount = icount + 1
    #end of data line loop 
    
    # get rid of filled values where nans are present in input data
    bad_input = np.where(np.isnan(sw_dn) == 1)
    warm_output[bad_input,:] = np.nan
    
    # Recompute entire time series of fluxes with seawater T adjusted for warm layer
    del Bx
    Tsea = Tsea + warm_output[:,2]

    if out=='lim':
        Bx = c36(U,Zu,Tair,Zt,RH,Zq,P,Tsea,SW_dn,LW_dn,Lat,Lon,Jd,
                 Zi,Rainrate,Ss,cp,sigH,zrf_u,zrf_t,zrf_q,out='lim')
    if out=='full':
        Bx = c36(U,Zu,Tair,Zt,RH,Zq,P,Tsea,SW_dn,LW_dn,Lat,Lon,Jd,
                 Zi,Rainrate,Ss,cp,sigH,zrf_u,zrf_t,zrf_q,out='full')
    
    B = np.hstack((Bx,warm_output))
    return B
    
  
def scalarv(P0 = None,tsea = None,tair = None,rh = None,zt = None): 
    # Compute the require scalar variables for the bulk code
    # Vectorized when needed
    # Inputs:
    # P0    Air pressure (mb)
    # tsea  sea temp (C)
    # tair  Air temperature (C)
    # rh    Relative humidity (#)
    
    Press = P0 * 100
    P_tq = 100 * (P0 - (0.125 * zt))
    tseak = tsea + 273.15
    tairk = tair + 273.15
    
    if np.size(tsea) > 1:
        #**********************COMPUTES QSEA***************************
        Exx = 6.1121 * np.exp(17.502 * tsea / (240.97 + tsea))
        Exx = np.multiply(Exx,(1.0007 + P0 * 3.46e-06))
        Esatsea = Exx * 0.98
        Qsatsea = 0.622 * Esatsea / (P0 - 0.378 * Esatsea) * 1000
        #**********************COMPUTES QAIR***************************
        Exx = 6.1121 * np.exp(17.502 * tair / (240.97 + tair))
        Exx = np.multiply(Exx,(1.0007 + P_tq * 3.46e-06))
        Qsatms = 0.622 * Exx / (P_tq - 0.378 * Exx) * 1000
        Ems = np.multiply(Exx,rh) / 100
        Qms = 0.622 * Ems / (P_tq - 0.378 * Ems) * 1000
        E = Ems * 100
        #******************COMPUTES AIR DENSITY*******************
        Rhoair = P_tq / (np.multiply(tairk,(1 + 0.61 * Qms / 1000)) * 287.05)
        Rhodry = (P_tq - E) / (tairk * 287.05)
    else:
        #**********************COMPUTES QSEA***************************
        Ex = ComputeEsat(tsea,P0)
        Esatsea = Ex * 0.98
        Qsatsea = 0.622 * Esatsea / (P0 - 0.378 * Esatsea) * 1000
        #**********************COMPUTES QAIR***************************
        Esatms = ComputeEsat(tair,P_tq)
        Qsatms = 0.622 * Esatms / (P_tq - 0.378 * Esatms) * 1000
        Ems = Esatms * rh / 100
        Qms = 0.622 * Ems / (P_tq - 0.378 * Ems) * 1000
        E = Ems * 100
        #******************COMPUTES AIR DENSITY*******************
        Rhoair = P_tq / (tairk * (1 + 0.61 * Qms / 1000) * 287.05)
        Rhodry = (P_tq - E) / (tairk * 287.05)
    
    return Press,tseak,tairk,Qsatsea,Qsatms,Qms,Rhoair,Rhodry
    

def ComputeEsat(T = None,P = None): 
    #   Given temperature (C) and pressure (mb), returns
    #   saturation vapor pressure (mb).
    Exx = 6.1121 * np.exp(17.502 * T / (240.97 + T))
    Exx = np.multiply(Exx,(1.0007 + P * 3.46e-06))
    return Exx

    
def psit_26_3p6(zeta = None): 
    # computes temperature structure function
    dzeta = np.minimum(50,0.35 * zeta)
    psi = - ((1 + 0.6667 * zeta) ** 1.5 + np.multiply(0.6667 * (zeta - 14.28),np.exp(- dzeta)) + 8.525)
    k = np.array(np.where(zeta < 0))
    x = (1 - 15 * zeta[k]) ** 0.5
    psik = 2 * np.log((1 + x) / 2)
    x = (1 - 34.15 * zeta[k]) ** 0.3333
    psic = (1.5 * np.log((1 + x + x ** 2) / 3) - np.sqrt(3) * np.arctan((1 + 2 * x) / np.sqrt(3)) 
            + 4 * np.arctan(1) / np.sqrt(3))
    f = zeta[k] ** 2.0 / (1 + zeta[k] ** 2)
    psi[k] = np.multiply((1 - f),psik) + np.multiply(f,psic)
    return psi
    
    
def psiu_26_3p6(zeta = None): 
    # computes velocity structure function
    dzeta = np.minimum(50,0.35 * zeta)
    a = 0.7
    b = 3 / 4
    c = 5
    d = 0.35
    psi = - (a * zeta + np.multiply(b * (zeta - c / d),np.exp(- dzeta)) + b * c / d)
    k = np.array(np.where(zeta < 0))
    x = (1 - 15 * zeta[k]) ** 0.25
    psik = (2 * np.log((1 + x) / 2) + np.log((1 + np.multiply(x,x)) / 2) 
            - 2 * np.arctan(x) + 2 * np.arctan(1))
    x = (1 - 10.15 * zeta[k]) ** 0.3333
    psic = (1.5 * np.log((1 + x + x ** 2) / 3) - np.sqrt(3) * np.arctan((1 + 2 * x) / np.sqrt(3)) 
            + 4 * np.arctan(1) / np.sqrt(3))
    f = zeta[k] ** 2.0 / (1 + zeta[k] ** 2)
    psi[k] = np.multiply((1 - f),psik) + np.multiply(f,psic)
    return psi
    
    
def psiu_40_3p6(zeta = None): 
    # computes velocity structure function
    dzeta = np.minimum(50,0.35 * zeta)
    a = 1
    b = 3 / 4
    c = 5
    d = 0.35
    psi = - (a * zeta + np.multiply(b * (zeta - c / d),np.exp(- dzeta)) + b * c / d)
    k = np.array(np.where(zeta < 0))
    x = (1 - 18 * zeta[k]) ** 0.25
    psik = (2 * np.log((1 + x) / 2) + np.log((1 + np.multiply(x,x)) / 2) 
            - 2 * np.arctan(x) + 2 * np.arctan(1))
    x = (1 - 10 * zeta[k]) ** 0.3333
    psic = (1.5 * np.log((1 + x + x ** 2) / 3) - np.sqrt(3) * np.arctan((1 + 2 * x) / np.sqrt(3)) 
            + 4 * np.arctan(1) / np.sqrt(3))
    f = zeta[k] ** 2.0 / (1 + zeta[k] ** 2)
    psi[k] = np.multiply((1 - f),psik) + np.multiply(f,psic)
    return psi
    
    
def bucksat(T = None,P = None,Tf = None): 
    # computes saturation vapor pressure [mb]
    # given T [degC] and P [mb] Tf is freezing pt
    exx = np.multiply(np.multiply(6.1121,np.exp(np.multiply(17.502,T) / (T + 240.97))),
                      (1.0007 + np.multiply(3.46e-06,P)))
    ii = np.array(np.where(T < Tf))
    if np.size(ii) != 0:
        exx[ii] = np.multiply(np.multiply((1.0003 + 4.18e-06 * P[ii]),6.1115),
                              np.exp(np.multiply(22.452,T[ii]) / (T[ii] + 272.55)))
    return exx
    
    
def qsat26sea(T = None,P = None,Ss = None,Tf = None): 
    # computes surface saturation specific humidity [g/kg]
    # given T [degC] and P [mb]
    ex = bucksat(T,P,Tf)
    fs = 1 - 0.02 * Ss / 35
    es = np.multiply(fs,ex)
    qs = 622 * es / (P - 0.378 * es)
    return qs
    
    
def qsat26air(T = None,P = None,rh = None): 
    # computes saturation specific humidity [g/kg]
    # given T [degC] and P [mb]
    Tf = 0
    es = bucksat(T,P,Tf)
    em = np.multiply(0.01 * rh,es)
    q = 622 * em / (P - 0.378 * em)
    return q,em
    
    
def grv(lat = None): 
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
        

def rhcalc_3p5(t,p,q):
    """
    usage: rh = rhcalc_3p5(t,p,q)
    Returns RH(%) for given t(C), p(mb) and specific humidity, q(kg/kg)

    Returns ndarray float for any numeric object input.
    """
    q2 = np.copy(np.asarray(q, dtype=float))    # conversion to ndarray float
    p2 = np.copy(np.asarray(p, dtype=float))
    t2 = np.copy(np.asarray(t, dtype=float))
    es = qsat_3p5(t2,p2)
    em = p2 * q2 / (0.622 + 0.378 * q2)
    rh = 100.0 * em / es
    return rh


def rhcalc_3p6(T = None,P = None,Q = None,Tf = None): 
    # computes relative humidity given T,P, & Q
    es = np.multiply(np.multiply(6.1121,np.exp(np.multiply(17.502,T) / (T + 240.97))),(1.0007 + np.multiply(3.46e-06,P)))
    ii = np.array(np.where(T < Tf))
    if np.size(ii) != 0:
        es[ii] = np.multiply(np.multiply(6.1115,np.exp(np.multiply(22.452,T[ii]) / (T[ii] + 272.55))),(1.0003 + 4.18e-06 * P[ii]))
    em = np.multiply(Q,P) / (np.multiply(0.378,Q) + 0.622)
    RHrf = 100 * em / es
    return RHrf
        

def albedo_vector(sw_dn = None,jd = None,lon = None,lat = None,eorw = None): 
    #  Computes transmission and albedo from downwelling sw_dn using
    #  lat   : latitude in degrees (positive to the north)
    #  lon   : longitude in degrees (positive to the west)
    #  jd    : yearday
    #  sw_dn : downwelling solar radiation measured at surface
    #  eorw  : 'E' if longitude is positive to the east, or 'W' if otherwise
        
    # updates:
    #   20-10-2021: ET vectorized function
    
    if eorw == 'E':
        lon = - lon
    elif eorw == 'W':
        pass
    else:
        print('please provide sign information on whether lon is deg E or deg W')
        
    
    alb = np.full([np.size(sw_dn)],np.nan)
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    SC = 1380
    utc = (jd - np.fix(jd)) * 24
    h = np.pi * utc / 12 - lon
    declination = 23.45 * np.cos(2 * np.pi * (jd - 173) / 365.25)
    solarzenithnoon = (lat * 180 / np.pi - declination)
    solaraltitudenoon = 90 - solarzenithnoon
    sd = declination * np.pi / 180
    gamma = 1
    gamma2 = gamma * gamma
    
    sinpsi = np.multiply(np.sin(lat),np.sin(sd)) - np.multiply(np.multiply(np.cos(lat),np.cos(sd)),np.cos(h))
    psi = np.multiply(np.arcsin(sinpsi),180) / np.pi
    solarmax = np.multiply(SC,sinpsi) / gamma2
    
    T = np.minimum(2,sw_dn / solarmax)    
    Ts = np.arange(0,1+0.05,0.05)
    As = np.arange(0,90+2,2)
    
    #  Look up table from Payne (1972)  Only adjustment is to T=0.95 Alt=10 value
    a = np.array([[ 0.061,0.061,0.061,0.061,0.061,0.061,0.061,0.061,
                    0.061,0.061,0.061,0.061,0.061,0.061,0.061,0.061,
                    0.061,0.061,0.061,0.061,0.061,0.061,0.061,0.061,
                    0.061,0.061,0.061,0.061,0.061,0.061,0.061,0.061,
                    0.061,0.061,0.061,0.061,0.061,0.061,0.061,0.061,
                    0.061,0.061,0.061,0.061,0.061,0.061],
                   [0.062,0.062,0.061,0.061,0.061,0.061,0.061,0.061,
                    0.061,0.061,0.061,0.061,0.061,0.061,0.061,0.061,
                    0.061,0.061,0.061,0.061,0.061,0.061,0.061,0.061,
                    0.061,0.061,0.061,0.061,0.061,0.061,0.061,0.061,
                    0.061,0.061,0.061,0.061,0.061,0.061,0.061,0.061,
                    0.061,0.061,0.061,0.061,0.061,0.061],
                   [0.072,0.070,0.068,0.065,0.065,0.063,0.062,0.061,
                    0.061,0.061,0.061,0.061,0.061,0.061,0.061,0.061,
                    0.060,0.061,0.060,0.060,0.060,0.060,0.060,0.060,
                    0.060,0.060,0.060,0.060,0.060,0.060,0.060,0.060,
                    0.060,0.060,0.060,0.060,0.060,0.060,0.060,0.060,
                    0.060,0.060,0.060,0.060,0.060,0.060],
                   [0.087,0.083,0.079,0.073,0.070,0.068,0.066,0.065,
                    0.064,0.063,0.062,0.061,0.061,0.060,0.060,0.060,
                    0.060,0.060,0.060,0.060,0.060,0.060,0.060,0.060,
                    0.060,0.060,0.060,0.060,0.060,0.060,0.060,0.060,
                    0.060,0.060,0.060,0.060,0.060,0.060,0.060,0.060,
                    0.060,0.060,0.060,0.060,0.060,0.060],
                   [0.115,0.108,0.098,0.086,0.082,0.077,0.072,0.071,
                    0.067,0.067,0.065,0.063,0.062,0.061,0.061,0.060,
                    0.060,0.060,0.060,0.061,0.061,0.061,0.061,0.060,
                    0.059,0.060,0.060,0.060,0.060,0.060,0.060,0.060,
                    0.060,0.060,0.060,0.060,0.060,0.060,0.060,0.060,
                    0.060,0.060,0.060,0.059,0.059,0.059],
                   [0.163,0.145,0.130,0.110,0.101,0.092,0.084,0.079,
                    0.072,0.072,0.068,0.067,0.064,0.063,0.062,0.061,
                    0.061,0.061,0.060,0.060,0.060,0.060,0.060,0.059,
                    0.059,0.059,0.059,0.059,0.059,0.059,0.059,0.059,
                    0.059,0.059,0.059,0.059,0.059,0.059,0.059,0.059,
                    0.059,0.059,0.059,0.059,0.059,0.058],
                   [0.235,0.198,0.174,0.150,0.131,0.114,0.103,0.094,
                    0.083,0.080,0.074,0.074,0.070,0.067,0.065,0.064,
                    0.063,0.062,0.061,0.060,0.060,0.060,0.059,0.059,
                    0.059,0.059,0.059,0.059,0.059,0.059,0.059,0.059,
                    0.059,0.059,0.059,0.059,0.059,0.059,0.059,0.059,
                    0.059,0.059,0.059,0.058,0.058,0.058],
                   [0.318,0.263,0.228,0.192,0.168,0.143,0.127,0.113,
                    0.099,0.092,0.084,0.082,0.076,0.072,0.070,0.067,
                    0.065,0.064,0.062,0.062,0.060,0.060,0.060,0.059,
                    0.059,0.059,0.059,0.059,0.059,0.059,0.058,0.058,
                    0.058,0.058,0.058,0.058,0.058,0.058,0.057,0.058,
                    0.058,0.058,0.058,0.057,0.057,0.057],
                    [0.395,0.336,0.29,0.248,0.208,0.176,0.151,0.134,
                    0.117,0.107,0.097,0.091,0.085,0.079,0.075,0.071,
                    0.068,0.067,0.065,0.063,0.062,0.061,0.060,0.060,
                    0.060,0.059,0.059,0.058,0.058,0.058,0.057,0.057,
                    0.057,0.057,0.057,0.057,0.057,0.056,0.056,0.056,
                    0.056,0.056,0.056,0.056,0.056,0.055],
                   [0.472,0.415,0.357,0.306,0.252,0.210,0.176,0.154,
                    0.135,0.125,0.111,0.102,0.094,0.086,0.081,0.076,
                    0.072,0.071,0.068,0.066,0.065,0.063,0.062,0.061,
                    0.060,0.059,0.058,0.057,0.057,0.057,0.056,0.055,
                    0.055,0.055,0.055,0.055,0.055,0.054,0.053,0.054,
                    0.053,0.053,0.054,0.054,0.053,0.053],
                   [0.542,0.487,0.424,0.360,0.295,0.242,0.198,0.173,
                    0.150,0.136,0.121,0.110,0.101,0.093,0.086,0.081,
                    0.076,0.073,0.069,0.067,0.065,0.064,0.062,0.060,
                    0.059,0.058,0.057,0.056,0.055,0.055,0.054,0.053,
                    0.053,0.052,0.052,0.052,0.051,0.051,0.050,0.050,
                    0.050,0.050,0.051,0.050,0.050,0.050],
                   [0.604,0.547,0.498,0.407,0.331,0.272,0.219,0.185,
                    0.160,0.141,0.127,0.116,0.105,0.097,0.089,0.083,
                    0.077,0.074,0.069,0.066,0.063,0.061,0.059,0.057,
                    0.056,0.055,0.054,0.053,0.053,0.052,0.051,0.050,
                    0.050,0.049,0.049,0.049,0.048,0.047,0.047,0.047,
                    0.046,0.046,0.047,0.047,0.046,0.046],
                   [0.655,0.595,0.556,0.444,0.358,0.288,0.236,0.190,
                    0.164,0.145,0.130,0.119,0.107,0.098,0.090,0.084,
                    0.076,0.073,0.068,0.064,0.060,0.058,0.056,0.054,
                    0.053,0.051,0.050,0.049,0.048,0.048,0.047,0.046,
                    0.046,0.045,0.045,0.045,0.044,0.043,0.043,0.043,
                    0.042,0.042,0.043,0.042,0.042,0.042],
                   [0.693,0.631,0.588,0.469,0.375,0.296,0.245,0.193,
                    0.165,0.145,0.131,0.118,0.106,0.097,0.088,0.081,
                    0.074,0.069,0.065,0.061,0.057,0.055,0.052,0.050,
                    0.049,0.047,0.046,0.046,0.044,0.044,0.043,0.042,
                    0.042,0.041,0.041,0.040,0.040,0.039,0.039,0.039,
                    0.038,0.038,0.038,0.038,0.038,0.038],
                   [0.719,0.656,0.603,0.480,0.385,0.300,0.250,0.193,
                    0.164,0.145,0.131,0.116,0.103,0.092,0.084,0.076,
                    0.071,0.065,0.061,0.057,0.054,0.051,0.049,0.047,
                    0.045,0.043,0.043,0.042,0.041,0.040,0.039,0.039,
                    0.038,0.038,0.037,0.036,0.036,0.035,0.035,0.034,
                    0.034,0.034,0.034,0.034,0.034,0.034],
                   [0.732,0.670,0.592,0.474,0.377,0.291,0.246,0.190,
                    0.162,0.144,0.130,0.114,0.100,0.088,0.080,0.072,
                    0.067,0.062,0.058,0.054,0.050,0.047,0.045,0.043,
                    0.041,0.039,0.039,0.038,0.037,0.036,0.036,0.035,
                    0.035,0.034,0.033,0.032,0.032,0.032,0.031,0.031,
                    0.031,0.030,0.030,0.030,0.030,0.030],
                   [0.730,0.652,0.556,0.444,0.356,0.273,0.235,0.188,
                    0.160,0.143,0.129,0.113,0.097,0.086,0.077,0.069,
                    0.064,0.060,0.055,0.051,0.047,0.044,0.042,0.039,
                    0.037,0.035,0.035,0.035,0.034,0.033,0.033,0.032,
                    0.032,0.032,0.029,0.029,0.029,0.029,0.028,0.028,
                    0.028,0.028,0.027,0.027,0.028,0.028],
                   [0.681,0.602,0.488,0.386,0.320,0.252,0.222,0.185,
                    0.159,0.142,0.127,0.111,0.096,0.084,0.075,0.067,
                    0.062,0.058,0.054,0.050,0.046,0.042,0.040,0.036,
                    0.035,0.033,0.032,0.032,0.031,0.030,0.030,0.030,
                    0.030,0.029,0.027,0.027,0.027,0.027,0.026,0.026,
                    0.026,0.026,0.026,0.026,0.026,0.026],
                   [0.581,0.494,0.393,0.333,0.288,0.237,0.211,0.182,
                    0.158,0.141,0.126,0.110,0.095,0.083,0.074,0.066,
                    0.061,0.057,0.053,0.049,0.045,0.041,0.039,0.034,
                    0.033,0.032,0.031,0.030,0.029,0.028,0.028,0.028,
                    0.028,0.027,0.026,0.026,0.026,0.025,0.025,0.025,
                    0.025,0.025,0.025,0.025,0.025,0.025],
                   [0.453,0.398,0.342,0.301,0.266,0.226,0.205,0.180,
                    0.157,0.140,0.125,0.109,0.095,0.083,0.074,0.065,
                    0.061,0.057,0.052,0.048,0.044,0.040,0.038,0.033,
                    0.032,0.031,0.030,0.029,0.028,0.027,0.027,0.026,
                    0.026,0.026,0.025,0.025,0.025,0.025,0.025,0.025,
                    0.025,0.025,0.025,0.025,0.025,0.025],
                   [0.425,0.370,0.325,0.290,0.255,0.220,0.200,0.178,
                    0.157,0.140,0.122,0.108,0.095,0.083,0.074,0.065,
                    0.061,0.056,0.052,0.048,0.044,0.04 ,0.038,0.033,
                    0.032,0.031,0.030,0.029,0.028,0.027,0.026,0.026,
                    0.026,0.026,0.025,0.025,0.025,0.025,0.025,0.025,
                    0.025,0.025,0.025,0.025,0.025,0.025]])

    if T.size==1:   ### for single value function
        Tchk = np.abs(Ts - T)
        i = np.array(np.where(Tchk == Tchk.min()))
        if psi < 0:
            alb = np.array([0])
            solarmax = np.array([0])
            T = np.array([0])
            j = np.array([0])
            psi = np.array([0])
        else:
            Achk = np.abs(As - psi)
            j = np.array(np.where(Achk == Achk.min()))
            szj = j.shape
            if szj[0] > 0:
                alb = a[i,j].flatten()
            else:
                pass
    else:  ### for vectorized function
        for k in np.arange(0,np.size(sinpsi)).reshape(-1):
            Tchk = np.abs(Ts - T[k])
            i = np.array(np.where(Tchk == Tchk.min()))
            if psi[k] < 0:
                alb[k] = 0
                solarmax[k] = 0
                T[k] = 0
                j = 0
                psi[k] = 0
            else:
                Achk = np.abs(As - psi[k])
                j = np.array(np.where(Achk == Achk.min()))
                szj = j.shape
                if szj[0] > 0:
                    alb[k] = a[i,j]
                else:
                    pass

    return alb,T,solarmax,psi


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
        raise ValueError ('find: Input should be ndarray')
    if b.dtype != 'bool':
        raise ValueError ('find: Input should be boolean array')
    if b.ndim > 1:
        raise ValueError ('find: Input should be 1-D')

    F = b.size - np.sum(b)    # number of False in b
    idx = np.argsort(b)[F:]   # argsort puts True at the end, so select [F:]
    idx = np.sort(idx)        # be sure values in idx are ordered low to high

    return idx

def qsat_3p5(t,p):
    """
    usage: es = qsat_3p5(t,p)
    Returns saturation vapor pressure es (mb) given t(C) and p(mb).

    After Buck, 1981: J.Appl.Meteor., 20, 1527-1532

    Returns ndarray float for any numeric object input.
    """
    t2 = np.copy(np.asarray(t, dtype=float))  # convert to ndarray float
    p2 = np.copy(np.asarray(p, dtype=float))
    es = 6.1121 * np.exp(17.502 * t2 / (240.97 + t2))
    es = es * (1.0007 + p2 * 3.46e-6)
    return es


def qsea_3p5(sst,p):
    """
    usage: qs = qsea_3p5(sst,p)
    Returns saturation specific humidity (g/kg) at sea surface
    given sst(C) and p(mb) input of any numeric type.

    Returns ndarray float for any numeric object input.
    """
    ex = qsat_3p5(sst,p) # returns ex as ndarray float
    es = 0.98 * ex
    qs = 622 * es /(p - 0.378 * es)
    return qs


def qair_3p5(t,p,rh):
    """
    usage: qa, em = qair_3p5(t,p,rh)
    Returns specific humidity (g/kg) and partial pressure (mb)
    given t(C), p(mb) and rh(%).

    Returns ndarray float for any numeric object input.
    """
    rh2 = np.copy(np.asarray(rh,dtype=float))  # conversion to ndarray float
    rh2 /= 100.0                         # frational rh
    p2 = np.copy(np.asarray(p, dtype=float))
    t2 = np.copy(np.asarray(t, dtype=float))
    em = rh2 * qsat_3p5(t2,p2)
    qa = 621.97 * em / (p2 - 0.378 * em)
    return (qa, em)


def rhoa_calc(t,p,rh):
    """
    computes moist air density from temperature, pressure and RH

    usage: Ra = rhoa_calc(t,p,rh)

    inputs: t (deg C), p (mb or hPa) and rh

    output: Ra = moist air density in kg/m3

    """
    Md = 0.028964                       # mol wt dry air, kg/mole
    Mv = 0.018016                       # mol wt water, kg/mole
    Tk = t + 273.15                     # deg Kelvin
    Pa = p*100.0                        # Pascals
    Rgas = 8.314                        # in m3 Pa/mol K
    Pv = (rh/100.0)*qsat_3p5(t,p)*100.0     # H2O vapor pressure in Pa
    Pd = Pa - Pv                        # pressure dry air
    Ra = (Pd*Md + Pv*Mv)/(Rgas*Tk)      # moist air density
    return Ra


def rhod(t,p):
    """
    computes dry air density from temperature and pressure

    usage: Rd = rhod(t,p)

    inputs: t (deg C), and p (mb or hPa)

    output: Rd = dry air density in kg/m3

    """
    Rd = 287.058        # gas const for dry air in J/kg K
    tk = t+273.15       # deg Kelvin
    Pa = p*100          # Pascals
    Rdry = Pa/(Rd*tk)   # dry air density, kg/m3
    return Rd

def psit_26_3p5(z_L):
    """
    usage psi = psit_26_3p5(z_L)

    Computes the temperature structure function given z/L.
    """
    zet = np.copy(np.asarray(z_L, dtype=float))    # conversion to ndarray float
    dzet = 0.35*zet
    dzet[dzet>50] = 50.           # stable
    zet_stable = np.where(zet>=0,zet,np.nan)
    psi = - ((1 + 0.6667*zet_stable)**1.5 + 0.6667*(zet_stable - 14.28)*np.exp(-dzet) + 8.525)
    k = find(zet < 0)            # unstable
    x = (1 - 15*zet[k])**0.5
    psik = 2*np.log((1 + x)/2.)
    x = (1 - 34.15*zet[k])**0.3333
    psic = 1.5*np.log((1.+x+x**2)/3.) - np.sqrt(3)*np.arctan((1 + 2*x)/np.sqrt(3))
    psic += 4*np.arctan(1.)/np.sqrt(3.)
    f = zet[k]**2 / (1. + zet[k]**2.)
    psi[k] = (1-f)*psik + f*psic
    return psi


def psiu_26_3p5(z_L):
    """
    usage: psi = psiu_26_3p5(z_L)

    Computes velocity structure function given z/L
    """
    zet = np.copy(np.asarray(z_L, dtype=float))    # conversion to ndarray float
    dzet = 0.35*zet
    dzet[dzet>50] = 50.           # stable
    a = 0.7
    b = 3./4.
    c = 5.
    d = 0.35
    psi = -(a*zet + b*(zet - c/d)*np.exp(-dzet) + b*c/d)
    k = find(zet < 0)         # unstable
    x = (1 - 15*zet[k])**0.25
    psik = 2.*np.log((1.+x)/2.) + np.log((1.+x*x)/2.) - 2.*np.arctan(x) + 2.*np.arctan(1.)
    x = (1 - 10.15*zet[k])**0.3333
    psic = 1.5*np.log((1.+x+x**2)/3.) - np.sqrt(3.)*np.arctan((1.+2.*x)/np.sqrt(3.))
    psic += 4*np.arctan(1.)/np.sqrt(3.)
    f = zet[k]**2 / (1.+zet[k]**2)
    psi[k] = (1-f)*psik + f*psic
    return psi


def psiu_40_3p5(z_L):
    """
    usage: psi = psiu_40_3p5(z_L)

    Computes velocity structure function given z/L
    """
    zet = np.copy(np.asarray(z_L, dtype=float))    # conversion to ndarray float
    dzet = 0.35*zet
    dzet[dzet>50] = 50.           # stable
    a = 1.
    b = 3./4.
    c = 5.
    d = 0.35
    psi = -(a*zet + b*(zet - c/d)*np.exp(-dzet) + b*c/d)
    k = find(zet < 0)         # unstable
    x = (1. - 18.*zet[k])**0.25
    psik = 2.*np.log((1.+x)/2.) + np.log((1.+x*x)/2.) - 2.*np.arctan(x) + 2.*np.arctan(1.)
    x = (1. - 10.*zet[k])**0.3333
    psic = 1.5*np.log((1.+x+x**2)/3.) - np.sqrt(3.)*np.arctan((1.+2.*x)/np.sqrt(3.))
    psic += 4.*np.arctan(1.)/np.sqrt(3.)
    f = zet[k]**2 / (1.+zet[k]**2)
    psi[k] = (1-f)*psik + f*psic
    return psi



def Le_water(t, sal):
    """
    computes latent heat of vaporization for pure water and seawater
    reference:  M. H. Sharqawy, J. H. Lienhard V, and S. M. Zubair, Desalination
                and Water Treatment, 16, 354-380, 2010. (http://web.mit.edu/seawater/)
    validity: 0 < t < 200 C;   0 <sal <240 g/kg

    usage: Le_w, Le_sw = Le_water(t, sal)

    inputs: T in deg C
            sal in ppt

    output: Le_w, Le_sw in J/g (kJ/kg)

    """
    # polynomial constants
    a = [2.5008991412E+06, -2.3691806479E+03, 2.6776439436E-01,
        -8.1027544602E-03, -2.0799346624E-05]

    Le_w = a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4
    Le_sw = Le_w*(1 - 0.001*sal)
    return (Le_w/1000.0, Le_sw/1000.0)


def uv2spd_dir(u,v):
    """
    converts u, v meteorological wind components to speed/direction
    where u is velocity from N and v is velocity from E (90 deg)

    usage spd, dir = uv2spd_dir(u, v)

    """
    spd = np.zeros_like(u)
    dir = np.zeros_like(u)

    spd = np.sqrt(u**2 + v**2)
    dir = np.arctan2(v, u)*180.0/np.pi

    return (spd, dir)


def spd_dir2uv(spd,dir):
    """
    converts wind speed / direction to u, v meteorological wind components
    where u is velocity from N and v is velocity from E (90 deg)

    usage u, v = uv2spd_dir(spd, dir)

    """
    dir2 = (np.copy(dir) + 180.0)*np.pi/180.0
    s = np.sin(dir2)
    c = np.cos(dir2)
    v = -spd*s
    u = -spd*c

    return (u, v)
