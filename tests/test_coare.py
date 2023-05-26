"""
Tests for the functions used in the COARE 3.5 bulk calculations.
All expected data comes from output of MATLAB code found at
https://github.com/NOAA-PSL/COARE-algorithm/blob/master/Matlab/COARE3.5/coare35vn.m
The input data should cover most oceanographic conditions:
T=0->50C, P=950->1050mb, Rh=50->100%, Q=0.005->0.015kg/kg
The input data for the c35 algorithm was taken from the same repository.
All outputs are asserted to be in agreement with MATLAB code to 5 sig figs.
"""

import pandas as pd
import numpy as np
import os
import scipy.io as sio


def mat_to_df(path):
    """
    read mat files into dataframe - excludes header, version, and globals
    does NOT work on matlab file version > 7.2, for that need to use h5py
    see here for more info: https://stackoverflow.com/a/19340117
    """
    mat = sio.loadmat(path)
    exclude = ['__header__', '__version__', '__globals__']
    out = {k: mat[k].flatten()
           for k in filter(lambda x: x not in exclude, mat.keys())}
    out = pd.DataFrame(out)
    return out


def test_c35():
    from pycoare.coare import c35
    input_data = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), 'data/test_35_data_input_final.csv'
            )
        ).to_dict(orient='list')

    actual = c35(**input_data, out='default')
    expected = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), 'data/test_35_data_output_final.csv'
            )
        ).to_numpy()

    np.testing.assert_allclose(actual, expected, atol=0, rtol=10**-5)


def test_rhcalc():
    from pycoare.util import rhcalc
    input_data = mat_to_df(
        os.path.join(
            os.path.dirname(__file__), 'data/rh_calcs_input.mat'
            )
    )

    actual = rhcalc(**input_data)
    expected = mat_to_df(
        os.path.join(
            os.path.dirname(__file__), 'data/rh_calcs_output.mat'
        )
    )

    np.testing.assert_allclose(actual, expected.values.flatten(),
                               atol=0, rtol=10**-5)


def test_psi():
    from pycoare.util import psit_26, psiu_26, psiu_40
    input_data = mat_to_df(
        os.path.join(
            os.path.dirname(__file__), 'data/psi_calcs_data.mat'
            )
    )

    actual = psit_26(input_data.zet)
    expected = input_data.psit_26_out.values
    np.testing.assert_allclose(actual, expected, atol=0, rtol=10**-5)

    actual = psiu_26(input_data.zet)
    expected = input_data.psiu_26_out.values
    np.testing.assert_allclose(actual, expected, atol=0, rtol=10**-5)

    actual = psiu_40(input_data.zet)
    expected = input_data.psiu_40_out.values
    np.testing.assert_allclose(actual, expected, atol=0, rtol=10**-5)


def test_qsat():
    from pycoare.util import qsat, qsea, qair
    input_data = mat_to_df(
        os.path.join(
            os.path.dirname(__file__), 'data/qsat_calcs_input.mat'
            )
    )

    output_data = mat_to_df(
        os.path.join(
            os.path.dirname(__file__), 'data/qsat_calcs_output.mat'
            )
    )

    actual = qsat(input_data.t, input_data.p)
    expected = output_data.qsat.values
    np.testing.assert_allclose(actual, expected, atol=0, rtol=10**-5)

    actual = qsea(input_data.t, input_data.p)
    expected = output_data.qsat_sea.values
    np.testing.assert_allclose(actual, expected, atol=0, rtol=10**-5)

    actual = qair(**input_data)[0]
    expected = output_data.qsat_air.values
    np.testing.assert_allclose(actual, expected, atol=0, rtol=10**-5)


def test_find():
    from pycoare.util import find
    input_data = np.arange(-100, 100)
    condition = input_data > 0
    actual = find(condition)
    expected = np.arange(101, 200)
    np.testing.assert_equal(actual, expected)
