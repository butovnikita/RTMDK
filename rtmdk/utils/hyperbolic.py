"""rtmdk/utils/hyperbolic.py — Poincare ball model operations."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def poincare_dist(u: NDArray, v: NDArray, ball_radius: float = 0.85) -> float:
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    if u_norm >= ball_radius or v_norm >= ball_radius:
        u = u * (ball_radius - 1e-6) / max(u_norm, 1e-8)
        v = v * (ball_radius - 1e-6) / max(v_norm, 1e-8)
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
    delta = u - v
    sq_delta = np.sum(delta ** 2)
    # Bug #3 FIX: Use ball_radius^2 in denominator for non-unit ball
    # Standard formula for unit ball: denom = (1 - ||u||^2) * (1 - ||v||^2)
    # For ball_radius r: denom = (r^2 - ||u||^2) * (r^2 - ||v||^2) / r^2
    r_sq = ball_radius ** 2
    denom = ((r_sq - u_norm ** 2) * (r_sq - v_norm ** 2)) / max(r_sq, 1e-8)
    arg = 1 + 2 * sq_delta / max(denom, 1e-8)
    return float(np.arccosh(np.clip(arg, 1.0, None)))


def exp_map_poincare(tangent: NDArray, base: NDArray, ball_radius: float = 0.85) -> NDArray:
    base_norm = np.linalg.norm(base)
    if base_norm >= ball_radius:
        base = base * (ball_radius - 1e-6) / max(base_norm, 1e-8)
        base_norm = np.linalg.norm(base)
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-8:
        return base.copy()
    denom = 1.0 + base_norm ** 2
    factor = np.tanh(tangent_norm * 0.5) / max(base_norm, 1e-8)
    result = base + tangent * factor
    r = np.linalg.norm(result)
    if r >= ball_radius:
        result = result * (ball_radius - 1e-6) / max(r, 1e-8)
    return result.astype(np.float32)


def log_map_poincare(point: NDArray, base: NDArray, ball_radius: float = 0.85) -> NDArray:
    base_norm = np.linalg.norm(base)
    if base_norm >= ball_radius:
        base = base * (ball_radius - 1e-6) / max(base_norm, 1e-8)
        base_norm = np.linalg.norm(base)
    diff = point - base
    diff_norm = np.linalg.norm(diff)
    if diff_norm < 1e-8:
        return np.zeros_like(point)
    factor = 2.0 / (1.0 - base_norm ** 2)
    tangent = diff * factor
    return tangent.astype(np.float32)


def mobius_add(x: NDArray, y: NDArray, ball_radius: float = 0.85) -> NDArray:
    x2 = np.sum(x ** 2)
    y2 = np.sum(y ** 2)
    xy = np.dot(x, y)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = 1 + 2 * xy + x2 * y2
    result = num / max(den, 1e-8)
    r = np.linalg.norm(result)
    if r >= ball_radius:
        result = result * (ball_radius - 1e-6) / max(r, 1e-8)
    return result.astype(np.float32)
