# numerics/branch_selection.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TrialResult:
    candidate: Any # critical point
    score: float
    trial_in: Any
    trial_out: Any
    info: Dict[str, Any]


def default_is_valid_solver_result(sol: Any) -> bool:
    """
    Check for scipy-like attributes (matching rk45 solver)
    """
    if sol is None:
        return False
    if not hasattr(sol, "t") or not hasattr(sol, "y"):
        return False
    if len(sol.t) < 2:
        return False
    # Some solvers provide .success (scipy.solve_ivp). If present, respect it.
    if getattr(sol, "success", True) is False:
        return False
    return True


def trial_integrate_candidate(
    *,
    rhs: Callable[[float, np.ndarray], np.ndarray],
    solve_ode: Callable[[Callable, Tuple[float, float], List[float]], Any],
    candidate: Any,  # expects .rc, .uc, .slope
    r_inner: float,
    eps: float,
    delta: float,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Trial integrate just around rc to probe candidate slope behavior.

    delta is some small value greater than eps.
    """
    rc = float(candidate.rc)
    uc = float(candidate.uc)
    slope = float(candidate.slope)

    r0_out = rc * (1.0 + eps)
    u0_out = uc + slope * (r0_out - rc)

    r0_in = rc * (1.0 - eps)
    u0_in = uc + slope * (r0_in - rc)

    r1_out = rc * (1.0 + delta)
    r1_in_target = rc * (1.0 - delta)
    r1_in = max(float(r_inner) * (1.0 + eps), r1_in_target)

    trial_out = solve_ode(rhs, (r0_out, r1_out), [u0_out])
    trial_in = solve_ode(rhs, (r0_in, r1_in), [u0_in])

    info = {
        "eps": eps,
        "delta": delta,
        "r_inner": r_inner,
        "r0_out": r0_out,
        "u0_out": u0_out,
        "r1_out": r1_out,
        "r0_in": r0_in,
        "u0_in": u0_in,
        "r1_in": r1_in,
    }
    return trial_in, trial_out, info


def score_trial_monotonic(
    *,
    trial_in: Any,
    trial_out: Any,
    require_positive_u: bool = True,
) -> float:
    """
    Score a candidate by wind-like monotonicity near rc.

    - Outward probe should accelerate: u_out_end > u_out_start
    - Inward probe should decelerate toward Sun: u_in_end < u_in_start (since r decreases)
    """
    if not default_is_valid_solver_result(trial_out) or not default_is_valid_solver_result(trial_in):
        return np.inf

    u_out = np.asarray(trial_out.y[0], dtype=float)
    u_in = np.asarray(trial_in.y[0], dtype=float)

    if not np.all(np.isfinite(u_out)) or not np.all(np.isfinite(u_in)):
        return np.inf

    # If we require positive u, check both solutions. 
    if require_positive_u and (np.any(u_out <= 0) or np.any(u_in <= 0)):
        return np.inf

    # generate stating and ending values for comparison
    u_out_start, u_out_end = float(u_out[0]), float(u_out[-1])
    u_in_start, u_in_end = float(u_in[0]), float(u_in[-1])

    score = 0.0

    # penalties for failing monotonic preferences
    if u_out_end <= u_out_start:
        score += 100.0
    if u_in_end >= u_in_start:
        score += 100.0

    return score


def select_candidate_by_trial(
    *,
    rhs: Callable[[float, np.ndarray], np.ndarray],
    solve_ode: Callable[[Callable, Tuple[float, float], List[float]], Any],
    candidates: Iterable[Any],  # expects .rc, .uc, .slope  
    r_inner: float,
    eps: float = 1e-3,
    delta: float = 5e-2,
    scorer: Callable[..., float] = score_trial_monotonic,
    scorer_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, List[TrialResult]]:
    """
    Try each candidate slope via short trial integrations and pick the best-scoring one.
    Returns (best_candidate, diagnostics).
    """
    scorer_kwargs = scorer_kwargs or {}

    results: List[TrialResult] = []
    best_candidate: Any = None
    best_score = np.inf

    for cand in candidates:
        trial_in, trial_out, info = trial_integrate_candidate(
            rhs=rhs,
            solve_ode=solve_ode,
            candidate=cand,
            r_inner=r_inner,
            eps=eps,
            delta=delta,
        )
        score = float(scorer(trial_in=trial_in, trial_out=trial_out, **scorer_kwargs))

        results.append(TrialResult(
            candidate=cand,
            score=score,
            trial_in=trial_in,
            trial_out=trial_out,
            info=info,
        ))

        if score < best_score:
            best_score = score
            best_candidate = cand

    if best_candidate is None or not np.isfinite(best_score):
        raise RuntimeError(
            "Could not select a physical critical-point branch. "
            "All candidates failed trial checks."
        )

    return best_candidate, results
