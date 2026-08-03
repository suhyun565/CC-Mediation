"""
Distribution-based analysis of DMIS stage profiles.

A speaker's Likert profile -> a categorical distribution over the six DMIS
stages. The distribution is built by mapping each per-stage mean from [1, 7]
into a non-negative weight v_k - 1 in [0, 6] and normalizing to sum to 1.

For two speakers, distributions are combined into a single group-level
distribution (default: arithmetic mean). Pre/post group distributions are
compared with the Wasserstein-1 distance (Earth Mover's Distance) on the
ordinal stage scale Denial(0) -> Integration(5). A signed version is
reported so that ethnocentric -> ethnorelative shifts come out positive.

Auxiliary metrics: Jensen-Shannon divergence, total variation distance,
first-order stochastic dominance check.

References:
  Bennett (1986, 1993, 2011) DMIS stage descriptions.
  Paige et al. (2003) IDI weighted-mean scoring conventions.
  Villani (2008) Optimal Transport (Wasserstein distance).
"""

from __future__ import annotations


# --- release bootstrap: make shared/ and metrics/ importable ---
import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parent.parent
for _sub in ("shared", "metrics"):
    _p = str(_RELEASE_ROOT / _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---



import math
from typing import Optional

from dmis_items import STAGE_ORDER, COMPOSITE_WEIGHTS

EPS = 1e-12


# ---------------------------------------------------------------------------
# Profile -> categorical distribution
# ---------------------------------------------------------------------------

def profile_to_distribution(stage_means: dict[str, float]) -> Optional[list[float]]:
    """
    Convert per-stage Likert means in [1, 7] to a length-6 categorical
    probability distribution over (Denial, Defense, Minimization,
    Acceptance, Adaptation, Integration).

    Logic:
      - Subtract 1 from each Likert mean to get a non-negative weight in
        [0, 6]; a Likert mean of 1 ("strongly disagree") contributes 0.
      - Normalize so weights sum to 1.

    Returns None if all weights are zero (degenerate case: speaker
    answered '1' on every item).
    """
    weights = [max(0.0, stage_means.get(s, 0.0) - 1.0) for s in STAGE_ORDER]
    total = sum(weights)
    if total <= EPS:
        return None
    return [w / total for w in weights]


def combine_distributions(
    distributions: list[list[float]],
    method: str = "mean",
) -> Optional[list[float]]:
    """
    Combine per-speaker distributions into a single group-level distribution.

    method = 'mean'      : element-wise arithmetic mean (always sums to 1)
    method = 'geometric' : element-wise geometric mean, then normalized.
                           Emphasizes stages where BOTH speakers are aligned.
    """
    if not distributions:
        return None
    n_stages = len(distributions[0])
    if any(len(d) != n_stages for d in distributions):
        return None
    if method == "mean":
        combined = [
            sum(d[k] for d in distributions) / len(distributions)
            for k in range(n_stages)
        ]
    elif method == "geometric":
        combined = []
        for k in range(n_stages):
            log_sum = 0.0
            for d in distributions:
                log_sum += math.log(max(d[k], EPS))
            combined.append(math.exp(log_sum / len(distributions)))
        s = sum(combined)
        if s <= EPS:
            return None
        combined = [c / s for c in combined]
    else:
        raise ValueError(f"unknown method: {method}")
    s = sum(combined)
    if s <= EPS:
        return None
    return [c / s for c in combined]


# ---------------------------------------------------------------------------
# Distribution distances
# ---------------------------------------------------------------------------

def cdf(p: list[float]) -> list[float]:
    """Cumulative distribution. Stages are already in ordinal order."""
    out: list[float] = []
    acc = 0.0
    for v in p:
        acc += v
        out.append(acc)
    return out


# Paige et al. (2003) developmental score positions for the six DMIS stages.
# The ethnocentric stages (Denial, Defense, Minimization) sit at -3, -2, -1
# and the ethnorelative stages (Acceptance, Adaptation, Integration) at
# +1, +2, +3. The 2-unit gap between Minimization and Acceptance encodes the
# qualitative paradigm shift between ethnocentric and ethnorelative
# orientations that Bennett identifies as the most fundamental boundary in
# the model. Inter-stage gaps are therefore (1, 1, 2, 1, 1).
STAGE_POSITIONS: tuple[float, ...] = (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)


def expected_position(p: list[float]) -> float:
    """Weighted mean position of the distribution under Paige et al. (2003)
    developmental scoring. Range [-3, +3]: negative = ethnocentric, positive
    = ethnorelative."""
    return sum(pos * v for pos, v in zip(STAGE_POSITIONS, p))


def wasserstein1(p: list[float], q: list[float]) -> float:
    """
    Wasserstein-1 distance between two distributions on the Paige et al.
    (2003) developmental scale. Stages sit at fixed positions
        Denial=-3, Defense=-2, Minimization=-1,
        Acceptance=+1, Adaptation=+2, Integration=+3
    so the 2-unit gap between Minimization and Acceptance gives that
    transition twice the cost of any other inter-stage transition.

    For 1D distributions on a fixed (non-evenly-spaced) ordinal support,
    W_1 equals the integral of |F_p(x) - F_q(x)| dx, which on a discrete
    support reduces to a sum of CDF differences weighted by inter-point
    gaps. Returns a non-negative scalar; theoretical max is 6.0 when one
    distribution is a delta at Denial (-3) and the other is a delta at
    Integration (+3).
    """
    if len(p) != len(q):
        raise ValueError("distributions of different length")
    if len(p) != len(STAGE_POSITIONS):
        raise ValueError(
            f"distribution length {len(p)} does not match "
            f"STAGE_POSITIONS length {len(STAGE_POSITIONS)}"
        )
    cp = cdf(p)
    cq = cdf(q)
    total = 0.0
    for i in range(len(p) - 1):
        gap = STAGE_POSITIONS[i + 1] - STAGE_POSITIONS[i]
        total += abs(cp[i] - cq[i]) * gap
    return total


def signed_wasserstein1(p_pre: list[float], p_post: list[float]) -> float:
    """
    Signed Wasserstein-1: positive when post is more ethnorelative than pre.

    Sign comes from the difference in the expected developmental position
    under Paige et al. (2003) weights:
        E_post - E_pre   where E uses STAGE_POSITIONS = (-3, -2, -1, +1, +2, +3)

    Magnitude is the (Paige-weighted) unsigned Wasserstein-1 distance.
    Range: roughly [-6, +6], with 0 = no movement.
    """
    base = wasserstein1(p_pre, p_post)
    e_pre = expected_position(p_pre)
    e_post = expected_position(p_post)
    sign = 1.0 if (e_post - e_pre) >= 0 else -1.0
    return sign * base


def total_variation(p: list[float], q: list[float]) -> float:
    """TV distance, range [0, 1]."""
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def js_divergence(p: list[float], q: list[float]) -> float:
    """
    Jensen-Shannon divergence (base e). Range [0, ln 2]. Symmetric and
    always finite.
    """
    def kl(a: list[float], b: list[float]) -> float:
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > EPS:
                s += ai * math.log(ai / max(bi, EPS))
        return s

    m = [(a + b) / 2.0 for a, b in zip(p, q)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def first_order_dominance(p_post: list[float], p_pre: list[float]) -> Optional[str]:
    """
    Check whether p_post first-order stochastically dominates p_pre on the
    ethnorelative direction (i.e. p_post puts more mass on higher stages).

    Returns:
      "post_dominates"  if F_post(k) <= F_pre(k) for all k AND strictly less
                         for at least one k. Means: post has shifted toward
                         ethnorelative compared to pre (good mediation).
      "pre_dominates"   if the opposite (regression).
      "neither"         otherwise (the CDFs cross).
    """
    cpre = cdf(p_pre)
    cpost = cdf(p_post)
    diff = [cpost[i] - cpre[i] for i in range(len(p_pre) - 1)]
    if all(d <= EPS for d in diff) and any(d < -EPS for d in diff):
        return "post_dominates"
    if all(d >= -EPS for d in diff) and any(d > EPS for d in diff):
        return "pre_dominates"
    return "neither"


# ---------------------------------------------------------------------------
# Convenience: single function returning the full comparison dict
# ---------------------------------------------------------------------------

def compare_distributions(
    pre_distribution: list[float],
    post_distribution: list[float],
) -> dict:
    """
    Run all distribution-comparison metrics. Used downstream as the DMIS
    side of the mediation effectiveness score.

    Expected positions are reported under the Paige et al. (2003)
    developmental scoring scale (STAGE_POSITIONS = -3, -2, -1, +1, +2, +3),
    which encodes the qualitative paradigm shift between Minimization (-1)
    and Acceptance (+1). The Wasserstein-1 distance also uses these
    positions, so the M -> A transition has gap 2 (twice the cost of any
    other inter-stage transition).

    For backward compatibility and easier inspection of the raw stage
    membership, the simple integer-stage expected index (Denial=0 ...
    Integration=5) is also reported as `expected_stage_index_*`.
    """
    e_pre_paige = expected_position(pre_distribution)
    e_post_paige = expected_position(post_distribution)
    e_pre_idx = sum(k * v for k, v in enumerate(pre_distribution))
    e_post_idx = sum(k * v for k, v in enumerate(post_distribution))
    return {
        "expected_position_pre": e_pre_paige,
        "expected_position_post": e_post_paige,
        "expected_position_shift": e_post_paige - e_pre_paige,
        "expected_stage_index_pre": e_pre_idx,
        "expected_stage_index_post": e_post_idx,
        "expected_stage_index_shift": e_post_idx - e_pre_idx,
        # Backward-compatible aliases (older callers used these names; they
        # now refer to the Paige-weighted expected positions, which is the
        # primary signal under the DMIS measurement scale.)
        "expected_stage_pre": e_pre_paige,
        "expected_stage_post": e_post_paige,
        "expected_stage_shift": e_post_paige - e_pre_paige,
        "wasserstein_1": wasserstein1(pre_distribution, post_distribution),
        "signed_wasserstein_1": signed_wasserstein1(
            pre_distribution, post_distribution
        ),
        "total_variation": total_variation(pre_distribution, post_distribution),
        "js_divergence": js_divergence(pre_distribution, post_distribution),
        "stochastic_dominance": first_order_dominance(
            post_distribution, pre_distribution
        ),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Speaker A: stuck in Denial/Defense
    A_pre = {
        "Denial": 6.0, "Defense": 5.5, "Minimization": 4.0,
        "Acceptance": 2.0, "Adaptation": 1.5, "Integration": 1.0,
    }
    # Speaker B: similar but slightly less ethnocentric
    B_pre = {
        "Denial": 5.5, "Defense": 5.0, "Minimization": 4.5,
        "Acceptance": 2.5, "Adaptation": 2.0, "Integration": 1.5,
    }
    # After mediation: both shifted toward Acceptance / Adaptation
    A_post = {
        "Denial": 3.0, "Defense": 2.5, "Minimization": 3.5,
        "Acceptance": 5.5, "Adaptation": 4.5, "Integration": 2.5,
    }
    B_post = {
        "Denial": 2.5, "Defense": 2.0, "Minimization": 3.0,
        "Acceptance": 6.0, "Adaptation": 5.0, "Integration": 3.0,
    }

    a_pre_d = profile_to_distribution(A_pre)
    b_pre_d = profile_to_distribution(B_pre)
    a_post_d = profile_to_distribution(A_post)
    b_post_d = profile_to_distribution(B_post)

    print("Pre A:", [f"{x:.2f}" for x in a_pre_d])
    print("Pre B:", [f"{x:.2f}" for x in b_pre_d])
    pre = combine_distributions([a_pre_d, b_pre_d])
    post = combine_distributions([a_post_d, b_post_d])
    print("\nGroup pre:  ", [f"{x:.2f}" for x in pre])
    print("Group post: ", [f"{x:.2f}" for x in post])

    cmp = compare_distributions(pre, post)
    for k, v in cmp.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")