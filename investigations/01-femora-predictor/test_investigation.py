import math

import numpy as np

import grid_filter as gf
from femora_port import FemoraPredictor, Params, java_round, steady_state_gain


def test_java_round():
    assert java_round(2.5) == 3 and java_round(-2.5) == -2 and java_round(2.4999) == 2


def test_kalman_matches_hand_computation():
    p = FemoraPredictor(28, feedback_mode="off")
    p.observe(30)
    # P- = 9.5, K = 9.5/13.5
    k = 9.5 / 13.5
    assert math.isclose(p.s.mu_kf, 28 + k * 2)
    assert math.isclose(p.s.p_kf, (1 - k) * 9.5)
    # EWMA always initialised: 0.3*30 + 0.7*28
    assert math.isclose(p.s.ewma_mean, 28.6)
    assert math.isclose(p.s.ewma_var, 0.7 * (9 + 0.3 * 4))
    assert math.isclose(p.s.mu, 0.5 * p.s.mu_kf + 0.5 * 28.6)
    assert math.isclose(p.s.sigma2, p.s.p_kf + p.s.ewma_var)


def test_steady_state_gain_is_fixed_point():
    k, P = steady_state_gain(0.5, 4.0)
    pp = P + 0.5
    assert math.isclose(pp / (pp + 4.0), k, rel_tol=1e-12)
    assert abs(k - 0.29654) < 1e-4


def test_gate_is_absorbing_after_level_shift():
    p = FemoraPredictor(28, feedback_mode="off")
    for _ in range(10):
        p.observe(28)            # tight history shrinks the EWMA variance
    for _ in range(20):
        assert p.observe(36) is False    # an 8-day shift is never accepted again
    assert java_round(p.s.mu) == 28


def test_as_written_feedback_never_persists():
    p = FemoraPredictor(28, feedback_mode="as_written")
    for z in [28, 33, 24, 33, 23, 34]:
        p.predict(); p.observe(z)
    assert p.s.sigma_scale == 1.0 and p.s.params.alpha == 0.3
    q = FemoraPredictor(28, feedback_mode="as_intended")
    for z in [28, 33, 24, 33, 23, 34]:
        q.predict(); q.observe(z)
    assert q.s.sigma_scale != 1.0 or q.s.params.alpha != 0.3


def test_grid_pmf_normalised_and_robust_to_missed_log():
    f = gf.GridFilter(29)
    for z in [29, 28, 30, 29]:
        f.update(z)
    before = gf.interval_from_pmf(f.predictive_pmf())
    f.update(58)                 # a missed log looks like a double cycle
    after = gf.interval_from_pmf(f.predictive_pmf())
    assert math.isclose(f.predictive_pmf().sum(), 1.0, rel_tol=1e-9)
    assert abs(after[2] - before[2]) <= 1


def test_hazard_and_conditioning():
    pmf = np.zeros(len(gf.DAYS)); pmf[27] = 0.5; pmf[29] = 0.5   # days 28 and 30
    h = gf.hazard(pmf)
    assert math.isclose(h[27], 0.5) and math.isclose(h[29], 1.0)
    c = gf.conditional_on_elapsed(pmf, 28)
    assert math.isclose(c[29], 1.0)
