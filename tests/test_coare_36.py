"""Tests for the COARE functions in pycoare.coare_36"""

import csv
import os

import numpy as np
import pytest

from pycoare import coare_36

# required or tests fail since numpy doesn't save enough precision
np.set_printoptions(precision=12)


@pytest.fixture
def load_input():
    path = os.path.join(os.path.dirname(__file__), "data/c35_c36_test_input.csv")
    data = list(csv.reader(open(path)))
    input_data = {i[0]: np.array(i[1:], dtype=float) for i in zip(*data)}
    input_data.update({"jcool": 1, "nits": 10})
    return input_data


@pytest.fixture
def load_expected():
    path = os.path.join(os.path.dirname(__file__), "data/c36_test_expected.csv")
    data = list(csv.reader(open(path)))
    expected = {i[0]: np.array(i[1:], dtype=float) for i in zip(*data)}
    return expected


class TestOutputC36:
    def test_inputs(self, load_input):
        expected = load_input
        actual = coare_36(**load_input)._bulk_loop_inputs  # noqa: SLF001
        np.testing.assert_allclose(actual.u, expected["u"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.t, expected["t"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.rh, expected["rh"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.zu, expected["zu"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.zt, expected["zt"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.zq, expected["zq"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.zrf, expected["zrf"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.us, expected["us"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.ts, expected["ts"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.p, expected["p"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.lat, expected["lat"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.zi, expected["zi"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.rs, expected["rs"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.rl, expected["rl"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.rain, expected["rain"], atol=1e-10, rtol=0)

    def test_fluxes(self, load_input, load_expected):
        expected = load_expected
        actual = coare_36(**load_input).fluxes
        np.testing.assert_allclose(actual.rnl, expected["rnl"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.tau, expected["tau"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.hsb, expected["hsb"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.hlb, expected["hlb"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.hbb, expected["hbb"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.hsbb, expected["hsbb"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            actual.hlwebb,
            expected["hlwebb"],
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_allclose(actual.evap, expected["evap"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.rf, expected["rf"], atol=1e-10, rtol=0)

    def test_velocities(self, load_input, load_expected):
        expected = load_expected
        actual = coare_36(**load_input).velocities
        np.testing.assert_allclose(actual.ut, expected["ut"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.usr, expected["usr"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.du, expected["du"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.gf, expected["gf"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.u, expected["u"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.u_rf, expected["u_rf"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.u_n, expected["u_n"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            actual.u_n_rf,
            expected["u_n_rf"],
            atol=1e-10,
            rtol=0,
        )

    def test_temperatures(self, load_input, load_expected):
        expected = load_expected
        actual = coare_36(**load_input).temperatures
        np.testing.assert_allclose(actual.lapse, expected["lapse"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.dt, expected["dt"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.dter, expected["dter"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.t_rf, expected["t_rf"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.t_n, expected["t_n"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            actual.t_n_rf,
            expected["t_n_rf"],
            atol=1e-10,
            rtol=0,
        )

    def test_humidities(self, load_input, load_expected):
        expected = load_expected
        actual = coare_36(**load_input).humidities
        np.testing.assert_allclose(actual.dq, expected["dq"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.dqer, expected["dqer"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.q_rf, expected["q_rf"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.q_n, expected["q_n"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            actual.q_n_rf,
            expected["q_n_rf"],
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_allclose(actual.rh_rf, expected["rh_rf"], atol=1e-10, rtol=0)

    def test_stability_parameters(self, load_input, load_expected):
        expected = load_expected
        actual = coare_36(**load_input).stability_parameters
        np.testing.assert_allclose(actual.tsr, expected["tsr"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.tvsr, expected["tvsr"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.tssr, expected["tssr"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.qsr, expected["qsr"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.tkt, expected["tkt"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.obukL, expected["obukL"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.zet, expected["zet"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.zo, expected["zo"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.zot, expected["zot"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.zoq, expected["zoq"], atol=1e-10, rtol=0)

    def test_transfer_coefficients(self, load_input, load_expected):
        expected = load_expected
        actual = coare_36(**load_input).transfer_coefficients
        np.testing.assert_allclose(actual.cd, expected["cd"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.ch, expected["ch"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.ce, expected["ce"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            actual.cdn_rf,
            expected["cdn_rf"],
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_allclose(
            actual.chn_rf,
            expected["chn_rf"],
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_allclose(
            actual.cen_rf,
            expected["cen_rf"],
            atol=1e-10,
            rtol=0,
        )

    def test_stability_functions(self, load_input, load_expected):
        expected = load_expected
        actual = coare_36(**load_input).stability_functions
        np.testing.assert_allclose(actual.psi_u, expected["psi_u"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            actual.psi_u_rf,
            expected["psi_u_rf"],
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_allclose(actual.psi_t, expected["psi_t"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            actual.psi_t_rf,
            expected["psi_t_rf"],
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_allclose(actual.psi_q, expected["psi_q"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            actual.psi_q_rf,
            expected["psi_q_rf"],
            atol=1e-10,
            rtol=0,
        )


class TestC36Attributes:
    def test_fluxes(self, load_input):
        c36 = coare_36(**load_input)
        assert hasattr(c36, "fluxes")

    def test_velocities(self, load_input):
        c36 = coare_36(**load_input)
        assert hasattr(c36, "velocities")

    def test_temperatures(self, load_input):
        c36 = coare_36(**load_input)
        assert hasattr(c36, "temperatures")

    def test_humidities(self, load_input):
        c36 = coare_36(**load_input)
        assert hasattr(c36, "humidities")

    def test_stability_parameters(self, load_input):
        c36 = coare_36(**load_input)
        assert hasattr(c36, "stability_parameters")

    def test_transfer_coefficients(self, load_input):
        c36 = coare_36(**load_input)
        assert hasattr(c36, "transfer_coefficients")

    def test_stability_functions(self, load_input):
        c36 = coare_36(**load_input)
        assert hasattr(c36, "stability_functions")


class TestSubclassAttributes:
    def test_fluxes(self, load_input):
        c36 = coare_36(**load_input)
        assert hasattr(c36.fluxes, "rnl")
        assert hasattr(c36.fluxes, "tau")
        assert hasattr(c36.fluxes, "hsb")
        assert hasattr(c36.fluxes, "hlb")
        assert hasattr(c36.fluxes, "hbb")
        assert hasattr(c36.fluxes, "hsbb")
        assert hasattr(c36.fluxes, "hlwebb")
        assert hasattr(c36.fluxes, "evap")
        assert hasattr(c36.fluxes, "rf")

    def test_velocities(self, load_input):
        c36 = coare_36(**load_input)
        assert hasattr(c36.velocities, "ut")
        assert hasattr(c36.velocities, "usr")
        assert hasattr(c36.velocities, "du")
        assert hasattr(c36.velocities, "gf")
        assert hasattr(c36.velocities, "u")
        assert hasattr(c36.velocities, "u_rf")
        assert hasattr(c36.velocities, "u_n")
        assert hasattr(c36.velocities, "u_n_rf")

    def test_temperatures(self, load_input):
        c36 = coare_36(**load_input)
        assert hasattr(c36.temperatures, "lapse")
        assert hasattr(c36.temperatures, "dt")
        assert hasattr(c36.temperatures, "dter")
        assert hasattr(c36.temperatures, "t_rf")
        assert hasattr(c36.temperatures, "t_n")
        assert hasattr(c36.temperatures, "t_n_rf")

    def test_humidities(self, load_input):
        c36 = coare_36(**load_input)
        assert hasattr(c36.humidities, "dq")
        assert hasattr(c36.humidities, "dqer")
        assert hasattr(c36.humidities, "q_rf")
        assert hasattr(c36.humidities, "q_n")
        assert hasattr(c36.humidities, "q_n_rf")
        assert hasattr(c36.humidities, "rh_rf")

    def test_stability_parameters(self, load_input):
        c36 = coare_36(**load_input)
        assert hasattr(c36.stability_parameters, "tsr")
        assert hasattr(c36.stability_parameters, "tvsr")
        assert hasattr(c36.stability_parameters, "tssr")
        assert hasattr(c36.stability_parameters, "qsr")
        assert hasattr(c36.stability_parameters, "tkt")
        assert hasattr(c36.stability_parameters, "obukL")
        assert hasattr(c36.stability_parameters, "zet")
        assert hasattr(c36.stability_parameters, "zo")
        assert hasattr(c36.stability_parameters, "zot")
        assert hasattr(c36.stability_parameters, "zoq")

    def test_transfer_coefficients(self, load_input):
        c36 = coare_36(**load_input)
        assert hasattr(c36.transfer_coefficients, "cd")
        assert hasattr(c36.transfer_coefficients, "ch")
        assert hasattr(c36.transfer_coefficients, "ce")
        assert hasattr(c36.transfer_coefficients, "cdn_rf")
        assert hasattr(c36.transfer_coefficients, "chn_rf")
        assert hasattr(c36.transfer_coefficients, "cen_rf")

    def test_stability_functions(self, load_input):
        c36 = coare_36(**load_input)
        assert hasattr(c36.stability_functions, "psi_u")
        assert hasattr(c36.stability_functions, "psi_u_rf")
        assert hasattr(c36.stability_functions, "psi_t")
        assert hasattr(c36.stability_functions, "psi_t_rf")
        assert hasattr(c36.stability_functions, "psi_q")
        assert hasattr(c36.stability_functions, "psi_q_rf")
