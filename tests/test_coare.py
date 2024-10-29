"""
Tests for the COARE functions in pycoare.util and pycoare.coare
"""

import csv
import numpy as np
import os
from pycoare.util import rhcalc, psit_26, psiu_26, psiu_40, qsat, qsea, qair
from pycoare import coare_35
import pytest

# required or tests fail since numpy doesn't save enough precision
np.set_printoptions(precision=12)


@pytest.fixture
def load_input():
    path = os.path.join(
        os.path.dirname(__file__), 'data/c35_test_input.csv'
    )
    data = list(csv.reader(open(path)))
    input_data = {i[0]: np.array(i[1:], dtype=float) for i in zip(*data)}
    input_data.update({'jcool': 1, 'nits': 10})
    return input_data


@pytest.fixture
def load_expected():
    path = os.path.join(
        os.path.dirname(__file__), 'data/c35_test_expected.csv'
    )
    data = list(csv.reader(open(path)))
    expected = {i[0]: np.array(i[1:], dtype=float) for i in zip(*data)}
    return expected


class TestOutputC35:
    def test_tau(self, load_input, load_expected):
        expected = load_expected['tau']
        actual = coare_35.tau(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_ustar(self, load_input, load_expected):
        expected = load_expected['ustar']
        actual = coare_35.ustar(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_tstar(self, load_input, load_expected):
        expected = load_expected['tstar']
        actual = coare_35.tstar(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_qstar(self, load_input, load_expected):
        expected = load_expected['qstar']
        actual = coare_35.qstar(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_sensible(self, load_input, load_expected):
        expected = load_expected['sensible']
        actual = coare_35.sensible(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_latent(self, load_input, load_expected):
        expected = load_expected['latent']
        actual = coare_35.latent(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_buoyancy(self, load_input, load_expected):
        expected = load_expected['buoyancy']
        actual = coare_35.buoyancy(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_webb(self, load_input, load_expected):
        expected = load_expected['webb']
        actual = coare_35.webb(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_cd(self, load_input, load_expected):
        expected = load_expected['cd']
        actual = coare_35.cd(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


class TestC35Attributes:
    def test_fluxes(self, load_input):
        c35 = coare_35(**load_input)
        assert hasattr(c35, 'fluxes')

    def test_velocities(self, load_input):
        c35 = coare_35(**load_input)
        assert hasattr(c35, 'velocities')

    def test_temperatures(self, load_input):
        c35 = coare_35(**load_input)
        assert hasattr(c35, 'temperatures')

    def test_humidities(self, load_input):
        c35 = coare_35(**load_input)
        assert hasattr(c35, 'humidities')

    def test_stability_parameters(self, load_input):
        c35 = coare_35(**load_input)
        assert hasattr(c35, 'stability_parameters')

    def test_transfer_coefficients(self, load_input):
        c35 = coare_35(**load_input)
        assert hasattr(c35, 'transfer_coefficients')

    def test_stability_functions(self, load_input):
        c35 = coare_35(**load_input)
        assert hasattr(c35, 'stability_functions')


class TestSubclassAttributes:
    def test_fluxes(self, load_input):
        c35 = coare_35(**load_input)
        assert hasattr(c35.fluxes, 'rnl')
        assert hasattr(c35.fluxes, 'tau')
        assert hasattr(c35.fluxes, 'hsb')
        assert hasattr(c35.fluxes, 'hlb')
        assert hasattr(c35.fluxes, 'hbb')
        assert hasattr(c35.fluxes, 'hsbb')
        assert hasattr(c35.fluxes, 'hlwebb')
        assert hasattr(c35.fluxes, 'evap')
        assert hasattr(c35.fluxes, 'rf')

    def test_velocities(self, load_input):
        c35 = coare_35(**load_input)
        assert hasattr(c35.velocities, 'ut')
        assert hasattr(c35.velocities, 'usr')
        assert hasattr(c35.velocities, 'du')
        assert hasattr(c35.velocities, 'gf')
        assert hasattr(c35.velocities, 'u')
        assert hasattr(c35.velocities, 'u_rf')
        assert hasattr(c35.velocities, 'u_n')
        assert hasattr(c35.velocities, 'u_n_rf')

    def test_temperatures(self, load_input):
        c35 = coare_35(**load_input)
        assert hasattr(c35.temperatures, 'lapse')
        assert hasattr(c35.temperatures, 'dt')
        assert hasattr(c35.temperatures, 'dter')
        assert hasattr(c35.temperatures, 't_rf')
        assert hasattr(c35.temperatures, 't_n')
        assert hasattr(c35.temperatures, 't_n_rf')

    def test_humidities(self, load_input):
        c35 = coare_35(**load_input)
        assert hasattr(c35.humidities, 'dq')
        assert hasattr(c35.humidities, 'dqer')
        assert hasattr(c35.humidities, 'q_rf')
        assert hasattr(c35.humidities, 'q_n')
        assert hasattr(c35.humidities, 'q_n_rf')
        assert hasattr(c35.humidities, 'rh_rf')

    def test_stability_parameters(self, load_input):
        c35 = coare_35(**load_input)
        assert hasattr(c35.stability_parameters, 'tsr')
        assert hasattr(c35.stability_parameters, 'tvsr')
        assert hasattr(c35.stability_parameters, 'tssr')
        assert hasattr(c35.stability_parameters, 'qsr')
        assert hasattr(c35.stability_parameters, 'tkt')
        assert hasattr(c35.stability_parameters, 'obukL')
        assert hasattr(c35.stability_parameters, 'zet')
        assert hasattr(c35.stability_parameters, 'zo')
        assert hasattr(c35.stability_parameters, 'zot')
        assert hasattr(c35.stability_parameters, 'zoq')

    def test_transfer_coefficients(self, load_input):
        c35 = coare_35(**load_input)
        assert hasattr(c35.transfer_coefficients, 'cd')
        assert hasattr(c35.transfer_coefficients, 'ch')
        assert hasattr(c35.transfer_coefficients, 'ce')
        assert hasattr(c35.transfer_coefficients, 'cdn_rf')
        assert hasattr(c35.transfer_coefficients, 'chn_rf')
        assert hasattr(c35.transfer_coefficients, 'cen_rf')

    def test_stability_functions(self, load_input):
        c35 = coare_35(**load_input)
        assert hasattr(c35.stability_functions, 'psi_u')
        assert hasattr(c35.stability_functions, 'psi_u_rf')
        assert hasattr(c35.stability_functions, 'psi_t')
        assert hasattr(c35.stability_functions, 'psi_t_rf')
        assert hasattr(c35.stability_functions, 'psi_q')
        assert hasattr(c35.stability_functions, 'psi_q_rf')


class TestUtil:
    @pytest.fixture
    def input_zet(self):
        return dict(z_L=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

    def test_psit_26(self, input_zet):
        expected = np.array(
            [
                3.12724961201,
                2.94563937069,
                2.7153981664,
                2.39730604321,
                1.86548667371,
                -0.0,
                -4.43410797233,
                -8.02103778406,
                -11.0874415242,
                -13.8543826804,
                -16.469041132,
            ]
        )
        actual = psit_26(**input_zet)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_psiu_26(self, input_zet):
        expected = np.array(
            [
                2.17508663882,
                2.0118078783,
                1.80735371734,
                1.53234530616,
                1.11049402203,
                -0.0,
                -4.39257224887,
                -7.53860684364,
                -9.85231262359,
                -11.6119662782,
                -13.0040743224,
            ]
        )
        actual = psiu_26(**input_zet)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_psiu_40(self, input_zet):
        expected = np.array(
            [
                2.16931171714,
                2.0088210855,
                1.80959461827,
                1.54608872607,
                1.15293343263,
                -0.0,
                -4.69257224887,
                -8.13860684364,
                -10.7523126236,
                -12.8119662782,
                -14.5040743224,
            ]
        )
        actual = psiu_40(**input_zet)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    @pytest.fixture()
    def input_rhcalc(self):
        return dict(
            t=[0, 1, 2, 3, 4],
            p=[1006, 1008, 1010, 1012, 1014],
            q=[0.001, 0.002, 0.003, 0.004, 0.005],
        )

    def test_rhcalc(self, input_rhcalc):
        expected = np.array(
            [26.335489852, 49.0630103047, 68.5938644747, 85.2938428107, 99.4883206669]
        )
        actual = rhcalc(**input_rhcalc)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    @pytest.fixture()
    def input_qsea(self):
        return dict(
            t=[0, 1, 2, 3, 4],
            p=[1006, 1008, 1010, 1012, 1014],
        )

    def test_qsat(self, input_qsea):
        expected = np.array(
            [6.1376532232, 6.59809274866, 7.08885199804, 7.61163590591, 8.1682276771]
        )
        actual = qsat(**input_qsea)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_qsea(self, input_qsea):
        expected = np.array(
            [3.72737831642, 3.99971180693, 4.28945019389, 4.59754176056, 4.92497736469]
        )
        actual = qsea(**input_qsea)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    @pytest.fixture()
    def input_qair(self):
        return dict(
            t=[0, 1, 2, 3, 4],
            p=[1006, 1008, 1010, 1012, 1014],
            rh=[65, 70, 75, 80, 85],
        )

    def test_qair(self, input_qair):
        expected = np.array(
            [2.47023726098, 2.85481658691, 3.28057712818, 3.7509893477, 4.26976278961]
        )
        actual = qair(**input_qair)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


class Testcoare_35:
    @pytest.fixture
    def load_input(self):
        path = os.path.join(os.path.dirname(__file__), "data/c35_test_input.csv")
        data = list(csv.reader(open(path)))
        input_data = {i[0]: np.array(i[1:], dtype=float) for i in zip(*data)}
        input_data.update({"jcool": 1, "nits": 10})
        return input_data

    @pytest.fixture
    def load_expected(self):
        path = os.path.join(os.path.dirname(__file__), "data/c35_test_expected.csv")
        data = list(csv.reader(open(path)))
        expected = {i[0]: np.array(i[1:], dtype=float) for i in zip(*data)}
        return expected

    def test_tau(self, load_input, load_expected):
        expected = load_expected["tau"]
        actual = coare_35.tau(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_ustar(self, load_input, load_expected):
        expected = load_expected["ustar"]
        actual = coare_35.ustar(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_tstar(self, load_input, load_expected):
        expected = load_expected["tstar"]
        actual = coare_35.tstar(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_qstar(self, load_input, load_expected):
        expected = load_expected["qstar"]
        actual = coare_35.qstar(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_sensible(self, load_input, load_expected):
        expected = load_expected["sensible"]
        actual = coare_35.sensible(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_latent(self, load_input, load_expected):
        expected = load_expected["latent"]
        actual = coare_35.latent(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_buoyancy(self, load_input, load_expected):
        expected = load_expected["buoyancy"]
        actual = coare_35.buoyancy(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_webb(self, load_input, load_expected):
        expected = load_expected["webb"]
        actual = coare_35.webb(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_cd(self, load_input, load_expected):
        expected = load_expected["cd"]
        actual = coare_35.cd(**load_input)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
