"""Faithful Python port of the FemoraAI CycleService predictor (Java, Aug 2025).

Source files mirrored (FemoraAI/CycleService, src/main/java/com/Femora/CycleService):
  service/prediction/engine/Kalman1DFilter.java
  service/prediction/engine/EwmaEstimator.java
  service/prediction/engine/OutlierDetector.java
  service/prediction/PredictionModelService.java   (updateOnNewPeriod, getPrediction)
  service/prediction/feedback/AutoEvaluator.java    (hit test, window of 8)
  service/prediction/feedback/AlphaTuner.java
  service/prediction/feedback/VarianceCalibrator.java
  service/prediction/ModelStateService.java         (defaults)

Excluded: the symptom personalization layer (DailyPredictor returns mu unchanged when no
symptom features are present, which is the case for synthetic data).

`feedback_mode`
  "as_written"  : reproduces the persistence order in the Java code. AutoEvaluator loads and
                  saves its own copy of ModelState; updateOnNewPeriod then saves the copy it
                  loaded earlier, overwriting alpha and sigmaScale. Net effect, the feedback
                  loop never persists. (Derived from code reading, Spring Data Mongo save()
                  replaces the whole document; not reproduced against a running service.)
  "as_intended" : the tuned alpha and sigmaScale are kept.
  "off"         : feedback disabled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

Z_TABLE = {0.80: 1.2815515655446004, 0.90: 1.6448536269514722, 0.95: 1.959963984540054}


def java_round(x: float) -> int:
    """java.lang.Math.round for doubles: floor(x + 0.5)."""
    return math.floor(x + 0.5)


@dataclass
class Params:
    alpha: float = 0.3
    z_cutoff: float = 3.0
    q: float = 0.5
    r: float = 4.0
    ci_level: float = 0.80
    w_kf: float = 0.5
    min_cycle: int = 15
    max_cycle: int = 60
    feedback_window: int = 8
    alpha_min: float = 0.1
    alpha_max: float = 0.6
    alpha_step: float = 0.05
    target_hit: float = 0.80
    ss_min: float = 0.8
    ss_max: float = 2.0
    ss_step: float = 0.1


@dataclass
class State:
    mu: float
    sigma2: float = 9.0
    ewma_mean: float = 0.0
    ewma_var: float = 9.0
    mu_kf: float = 0.0
    p_kf: float = 9.0
    sigma_scale: float = 1.0
    params: Params = field(default_factory=Params)


class FemoraPredictor:
    def __init__(self, init_days: float, feedback_mode: str = "as_written", params: Params | None = None):
        p = params or Params()
        self.s = State(mu=init_days, ewma_mean=init_days, mu_kf=init_days, params=p)
        self.mode = feedback_mode
        self.snapshot = None            # (lower_offset, upper_offset) of last issued prediction
        self.errors: list[bool] = []    # hit history, most recent last
        self.n_accepted = 0
        self.n_rejected = 0

    # ---- getPrediction --------------------------------------------------------------
    def predict(self) -> dict:
        s, p = self.s, self.s.params
        mu = s.mu
        sigma = math.sqrt(max(s.sigma2, 1e-6)) * max(0.1, min(5.0, s.sigma_scale))
        z = Z_TABLE.get(round(p.ci_level, 2), Z_TABLE[0.80])
        lo = java_round(mu - z * sigma)
        hi = java_round(mu + z * sigma)
        lo = max(p.min_cycle, min(p.max_cycle, lo))
        hi = max(p.min_cycle, min(p.max_cycle, hi))
        self.snapshot = (lo, hi)
        return {"expected": java_round(mu), "lower": lo, "upper": hi, "mu": mu, "sigma": sigma}

    def pmf(self, horizon: int = 90):
        """HazardService: Gaussian kernel at integer offsets 1..H, normalised."""
        s = self.s
        sigma = math.sqrt(max(s.sigma2, 1e-6)) * max(0.1, min(5.0, s.sigma_scale))
        w = [math.exp(-0.5 * ((d - s.mu) / sigma) ** 2) for d in range(1, horizon + 1)]
        tot = sum(w)
        return [x / tot for x in w]

    # ---- AutoEvaluator.evaluateOnActualStart ------------------------------------------
    def _evaluate(self, actual_offset: int, alpha_holder: Params, scale_holder: State):
        if self.snapshot is None:
            return
        lo, hi = self.snapshot
        self.errors.append(lo <= actual_offset <= hi)
        recent = self.errors[-8:][::-1]                     # findTop8 ... OrderByEvaluatedAtDesc
        hits = sum(recent)
        hit_rate = hits / len(recent)
        consec = 0
        for h in recent:
            if h:
                break
            consec += 1
        p = alpha_holder
        # AlphaTuner
        a = p.alpha
        if consec >= 2 or hit_rate < p.target_hit - 0.10:
            a = min(a + p.alpha_step, p.alpha_max)
        elif hit_rate >= p.target_hit + 0.05 and consec == 0:
            a = max(a - p.alpha_step, p.alpha_min)
        p.alpha = a
        # VarianceCalibrator
        cur = scale_holder.sigma_scale
        if hit_rate < p.target_hit:
            cur = min(cur + p.ss_step, p.ss_max)
        elif hit_rate > p.target_hit:
            cur = max(cur - p.ss_step, p.ss_min)
        scale_holder.sigma_scale = cur

    # ---- updateOnNewPeriod ------------------------------------------------------------
    def observe(self, days_since_prev_start: int) -> bool:
        """Returns True if the update was accepted (not gated as an outlier)."""
        s, p = self.s, self.s.params
        alpha_at_entry = p.alpha                                   # EwmaEstimator built first
        duration = max(p.min_cycle, min(p.max_cycle, days_since_prev_start))
        sd = math.sqrt(max(s.ewma_var, 1e-6))
        if abs((duration - s.ewma_mean) / sd) > p.z_cutoff:
            self.n_rejected += 1
            return False                                          # state unchanged, no evaluation

        if self.mode == "off":
            pass
        elif self.mode == "as_intended":
            self._evaluate(int(days_since_prev_start), p, s)
        else:  # as_written: evaluator mutates a separate copy that is later overwritten
            shadow_p = Params(**vars(p))
            shadow_s = State(**{**vars(s), "params": shadow_p})
            self._evaluate(int(days_since_prev_start), shadow_p, shadow_s)

        # EWMA (initialized=True because ewmaMean is always persisted)
        prev = s.ewma_mean
        s.ewma_mean = alpha_at_entry * duration + (1 - alpha_at_entry) * s.ewma_mean
        diff = duration - prev
        s.ewma_var = (1 - alpha_at_entry) * (s.ewma_var + alpha_at_entry * diff * diff)
        # Kalman
        p_pred = s.p_kf + p.q
        k = p_pred / (p_pred + p.r)
        s.mu_kf = s.mu_kf + k * (duration - s.mu_kf)
        s.p_kf = (1 - k) * p_pred
        # Blend
        s.mu = p.w_kf * s.mu_kf + (1 - p.w_kf) * s.ewma_mean
        s.sigma2 = s.p_kf + s.ewma_var
        self.n_accepted += 1
        return True


def steady_state_gain(q: float, r: float) -> tuple[float, float]:
    """Random walk plus noise: prior variance M solves M^2 - qM - qr = 0."""
    m = (q + math.sqrt(q * q + 4 * q * r)) / 2
    return m / (m + r), m - q
