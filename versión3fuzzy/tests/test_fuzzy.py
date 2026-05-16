"""
tests/test_fuzzy.py
--------------------
Unit tests for the fuzzy logic module.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from fuzzy.fuzzy_mission import (
    trimf,
    membership_low, membership_medium, membership_high,
    get_memberships,
    evaluate_rules,
    defuzzify_centroid,
    mission_health_index,
    evaluate_mission_state,
)


class TestMembershipFunctions:

    def test_trimf_at_peak(self):
        """At the peak, membership must be 1."""
        assert trimf(1.0, 0.0, 1.0, 2.0) == pytest.approx(1.0)

    def test_trimf_at_feet(self):
        """At the feet, membership must be 0."""
        assert trimf(0.0, 0.0, 1.0, 2.0) == pytest.approx(0.0)
        assert trimf(2.0, 0.0, 1.0, 2.0) == pytest.approx(0.0)

    def test_trimf_outside(self):
        """Outside the support, membership must be 0."""
        assert trimf(-1.0, 0.0, 1.0, 2.0) == 0.0
        assert trimf(3.0,  0.0, 1.0, 2.0) == 0.0

    def test_low_at_zero(self):
        """Level 0 must be fully LOW."""
        assert membership_low(0) == pytest.approx(1.0, abs=0.01)

    def test_low_at_two(self):
        """Level 2 must have zero LOW membership."""
        assert membership_low(2) == pytest.approx(0.0, abs=0.01)

    def test_medium_at_one(self):
        """Level 1 must be fully MEDIUM."""
        assert membership_medium(1) == pytest.approx(1.0, abs=0.01)

    def test_medium_at_extremes(self):
        """Level 0 and 2 must have zero MEDIUM membership."""
        assert membership_medium(0) == pytest.approx(0.0, abs=0.01)
        assert membership_medium(2) == pytest.approx(0.0, abs=0.01)

    def test_high_at_two(self):
        """Level 2 must be fully HIGH."""
        assert membership_high(2) == pytest.approx(1.0, abs=0.01)

    def test_high_at_zero(self):
        """Level 0 must have zero HIGH membership."""
        assert membership_high(0) == pytest.approx(0.0, abs=0.01)

    def test_get_memberships_returns_all_keys(self):
        """get_memberships must return low, medium and high keys."""
        m = get_memberships(1.0)
        assert "low"    in m
        assert "medium" in m
        assert "high"   in m

    def test_memberships_sum_to_one_at_level_1(self):
        """At level 1, low+medium+high approximately equals 1."""
        m = get_memberships(1.0)
        total = m["low"] + m["medium"] + m["high"]
        assert total == pytest.approx(1.0, abs=0.05)

    def test_all_memberships_in_range(self):
        """All membership values must be in [0, 1]."""
        for val in [0, 0.5, 1, 1.5, 2]:
            m = get_memberships(val)
            for k, v in m.items():
                assert 0.0 <= v <= 1.0, f"Membership {k}={v} out of range for val={val}"


class TestFuzzyRules:

    def test_all_critical_activates_critical_risk(self):
        """All resources at 0 must heavily activate CRITICAL risk."""
        activations = evaluate_rules(0, 0, 0, 0, 0)
        # CRITICAL key is 3.0
        assert activations[3.0] > 0.8

    def test_all_optimal_activates_safe(self):
        """All resources at 2 must heavily activate SAFE risk."""
        activations = evaluate_rules(2, 2, 2, 2, 2)
        assert activations[0.0] > 0.8

    def test_activations_in_range(self):
        """All activation values must be in [0, 1]."""
        for vals in [(0,0,0,0,0), (1,1,1,1,1), (2,2,2,2,2), (0,2,1,1,2)]:
            acts = evaluate_rules(*vals)
            for k, v in acts.items():
                assert 0.0 <= v <= 1.0


class TestDefuzzification:

    def test_defuzzify_returns_float(self):
        activations = {0.0: 0.8, 1.0: 0.1, 2.0: 0.0, 3.0: 0.0}
        result = defuzzify_centroid(activations)
        assert isinstance(result, float)

    def test_defuzzify_in_universe(self):
        """Defuzzified value must be in [0, 3]."""
        for vals in [(0,0,0,0,0), (1,1,1,1,1), (2,2,2,2,2)]:
            from fuzzy.fuzzy_mission import evaluate_rules
            acts = evaluate_rules(*vals)
            result = defuzzify_centroid(acts)
            assert 0.0 <= result <= 3.0

    def test_mhi_in_range(self):
        """MHI must be in [0, 1]."""
        for risk in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            mhi = mission_health_index(risk)
            assert 0.0 <= mhi <= 1.0

    def test_mhi_ordering(self):
        """Higher risk must correspond to lower MHI."""
        assert mission_health_index(0.0) > mission_health_index(1.5)
        assert mission_health_index(1.5) > mission_health_index(3.0)


class TestEvaluateMissionState:

    def test_returns_all_keys(self):
        result = evaluate_mission_state(1, 1, 1, 1, 1)
        for key in ["memberships", "activations", "crisp_risk", "mhi", "status"]:
            assert key in result

    def test_critical_state_has_low_mhi(self):
        """All resources at 0 must produce MHI < 0.25."""
        result = evaluate_mission_state(0, 0, 0, 0, 0)
        assert result["mhi"] < 0.25
        assert result["status"] == "CRITICAL"

    def test_optimal_state_has_high_mhi(self):
        """All resources at 2 must produce MHI > 0.75."""
        result = evaluate_mission_state(2, 2, 2, 2, 2)
        assert result["mhi"] > 0.75
        assert result["status"] == "OPTIMAL"

    def test_stable_state_has_medium_mhi(self):
        """All resources at 1 must produce MHI between 0.25 and 0.85."""
        result = evaluate_mission_state(1, 1, 1, 1, 1)
        assert 0.25 <= result["mhi"] <= 0.85

    def test_mhi_monotonic_with_resources(self):
        """Higher resources must produce higher MHI."""
        r0 = evaluate_mission_state(0, 0, 0, 0, 0)["mhi"]
        r1 = evaluate_mission_state(1, 1, 1, 1, 1)["mhi"]
        r2 = evaluate_mission_state(2, 2, 2, 2, 2)["mhi"]
        assert r0 < r1 < r2

    def test_energy_low_is_critical(self):
        """Energy at 0 with everything else stable must still be WARNING or CRITICAL."""
        result = evaluate_mission_state(0, 1, 1, 1, 1)
        assert result["status"] in ("CRITICAL", "WARNING")
