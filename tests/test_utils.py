"""Tests for the COARE functions in pycoare.utils"""

import numpy as np
import pytest

from pycoare.util import psit_26, psiu_26, psiu_40, qair, qsat, qsea, rhcalc

# required or tests fail since numpy doesn't save enough precision
np.set_printoptions(precision=12)


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
            ],
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
            ],
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
            ],
        )
        actual = psiu_40(**input_zet)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    @pytest.fixture
    def input_rhcalc(self):
        return dict(
            t=[0, 1, 2, 3, 4],
            p=[1006, 1008, 1010, 1012, 1014],
            q=[0.001, 0.002, 0.003, 0.004, 0.005],
        )

    def test_rhcalc(self, input_rhcalc):
        expected = np.array(
            [26.335489852, 49.0630103047, 68.5938644747, 85.2938428107, 99.4883206669],
        )
        actual = rhcalc(**input_rhcalc)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    @pytest.fixture
    def input_qsea(self):
        return dict(
            t=[0, 1, 2, 3, 4],
            p=[1006, 1008, 1010, 1012, 1014],
        )

    def test_qsat(self, input_qsea):
        expected = np.array(
            [6.1376532232, 6.59809274866, 7.08885199804, 7.61163590591, 8.1682276771],
        )
        actual = qsat(**input_qsea)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_qsea(self, input_qsea):
        expected = np.array(
            [3.72737831642, 3.99971180693, 4.28945019389, 4.59754176056, 4.92497736469],
        )
        actual = qsea(**input_qsea)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    @pytest.fixture
    def input_qair(self):
        return dict(
            t=[0, 1, 2, 3, 4],
            p=[1006, 1008, 1010, 1012, 1014],
            rh=[65, 70, 75, 80, 85],
        )

    def test_qair(self, input_qair):
        expected = np.array(
            [2.47023726098, 2.85481658691, 3.28057712818, 3.7509893477, 4.26976278961],
        )
        actual = qair(**input_qair)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
