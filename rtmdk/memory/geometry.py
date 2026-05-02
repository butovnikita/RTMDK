"""rtmdk/memory/geometry.py — Poincaré ball model operations.

All formulas are correct for arbitrary ball radius R (not just R=1).
Extracted from rtmdk/memory/core.py (Phase 5 architecture refactor).
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def _clip_norm(v: NDArray, max_norm: float) -> NDArray:
    norm = np.linalg.norm(v)
    if norm >= max_norm:
        return v * (max_norm - 1e-6) / max(norm, 1e-8)
    return v


def poincare_dist(u: NDArray, v: NDArray, ball_radius: float = 0.85) -> float:
    """Hyperbolic distance in Poincare ball model."""
    u = _clip_norm(u, ball_radius)
    v = _clip_norm(v, ball_radius)
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    delta = u - v
    sq_delta = np.sum(delta ** 2)
    r_sq = ball_radius ** 2
    denom = ((r_sq - u_norm ** 2) * (r_sq - v_norm ** 2)) / max(r_sq, 1e-8)
    arg = 1 + 2 * sq_delta / max(denom, 1e-8)
    return float(np.arccosh(np.clip(arg, 1.0, None)))


def exp_map_poincare(tangent: NDArray, base: NDArray, ball_radius: float = 0.85) -> NDArray:
    """Exponential map on Poincaré ball of radius R."""
    base = _clip_norm(base, ball_radius)
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-8:
        return base.copy().astype(np.float32)
    base_norm_sq = np.sum(base ** 2)
    lambda_base = 2.0 / (1.0 - base_norm_sq / (ball_radius ** 2))
    scaled_norm = lambda_base * tangent_norm / (2.0 * ball_radius)
    c = np.tanh(scaled_norm) / max(scaled_norm, 1e-8)
    direction = c * tangent
    result = mobius_add(base, direction, ball_radius)
    return result.astype(np.float32)


def log_map_poincare(point: NDArray, base: NDArray, ball_radius: float = 0.85) -> NDArray:
    """Logarithmic map on Poincaré ball of radius R."""
    base = _clip_norm(base, ball_radius)
    point = _clip_norm(point, ball_radius)
    diff = mobius_add(-base, point, ball_radius)
    diff_norm = np.linalg.norm(diff)
    if diff_norm < 1e-8:
        return np.zeros_like(point)
    base_norm_sq = np.sum(base ** 2)
    lambda_base = 2.0 / (1.0 - base_norm_sq / (ball_radius ** 2))
    ratio = diff_norm / ball_radius
    ratio = min(ratio, 1.0 - 1e-8)
    factor = (2.0 * ball_radius * np.arctanh(ratio)) / (lambda_base * diff_norm)
    tangent = diff * factor
    return tangent.astype(np.float32)


def mobius_add(x: NDArray, y: NDArray, ball_radius: float = 0.85) -> NDArray:
    """Möbius addition in Poincaré ball of radius R."""
    r_sq = ball_radius ** 2
    x2 = np.sum(x ** 2) / r_sq
    y2 = np.sum(y ** 2) / r_sq
    xy = np.dot(x, y) / r_sq
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = 1 + 2 * xy + x2 * y2
    result = num / max(den, 1e-8)
    return _clip_norm(result, ball_radius).astype(np.float32)
