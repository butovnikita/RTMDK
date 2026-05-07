"""Meta-cognitive controller for RTMDK."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    pass


class MetaController:
    def __init__(self,
                 n_trials: int = 20,
                 optimize_params: Optional[List[str]] = None,
                 optimization_freq: int = 500):
        self.n_trials = n_trials
        self.optimize_params = optimize_params or [
            "decay_rate", "tension_threshold", "phase_coupling", "bandwidth"
        ]
        self.optimization_freq = optimization_freq
        self._optuna_available = False
        self._best_params: Dict[str, float] = {}
        self._optimization_history: List[Dict] = []
        self._step_counter = 0
        self._last_optimization_time: float = 0.0
        self._total_optimizations = 0
        self._try_load_optuna()

    def _try_load_optuna(self):
        try:
            import optuna
            self._optuna_available = True
            self.optuna = optuna
        except ImportError:
            self._optuna_available = False

    def optimize(self, field: Any) -> Dict[str, float]:
        self._step_counter += 1
        if self._optuna_available:
            return self._optimize_with_optuna(field)
        else:
            return self._optimize_grid_search(field)

    def _optimize_with_optuna(self, field: Any) -> Dict[str, float]:
        def objective(trial):
            params = {}
            if "decay_rate" in self.optimize_params:
                params["decay_rate"] = trial.suggest_float(
                    "decay_rate", 0.95, 0.9999)
            if "tension_threshold" in self.optimize_params:
                params["tension_threshold"] = trial.suggest_float(
                    "tension_threshold", 0.1, 0.5)
            if "phase_coupling" in self.optimize_params:
                params["phase_coupling"] = trial.suggest_float(
                    "phase_coupling", 0.05, 0.8)
            if "bandwidth" in self.optimize_params:
                params["bandwidth"] = trial.suggest_float(
                    "bandwidth", 0.3, 5.0)
            return self._evaluate_params(field, params)

        study = self.optuna.create_study(
            direction="maximize",
            sampler=self.optuna.samplers.TPESampler(
                seed=42))
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=False)
        best_params = study.best_params
        self._best_params = best_params
        self._total_optimizations += 1
        self._last_optimization_time = time.time()
        self._optimization_history.append({
            "time": time.time(), "best_value": study.best_value,
            "params": best_params, "n_trials": self.n_trials,
        })
        return best_params

    def _optimize_grid_search(self, field: Any) -> Dict[str, float]:
        grid = {
            "decay_rate": [0.97, 0.98, 0.99, 0.995, 0.998],
            "tension_threshold": [0.15, 0.2, 0.25, 0.3, 0.35],
            "phase_coupling": [0.1, 0.2, 0.3, 0.4, 0.5],
            "bandwidth": [0.5, 1.0, 1.5, 2.0, 3.0],
        }
        filtered_grid = {
            k: v for k,
            v in grid.items() if k in self.optimize_params}
        best_score = -float("inf")
        best_params = {}
        keys = list(filtered_grid.keys())
        values = list(filtered_grid.values())
        n_trials = min(50, max(len(v) for v in values) ** len(keys))

        for _ in range(n_trials):
            params = {k: values[i][np.random.randint(
                len(values[i]))] for i, k in enumerate(keys)}
            score = self._evaluate_params(field, params)
            if score > best_score:
                best_score = score
                best_params = params.copy()

        self._best_params = best_params
        self._total_optimizations += 1
        self._last_optimization_time = time.time()
        self._optimization_history.append({
            "time": time.time(), "best_value": best_score,
            "params": best_params, "method": "random_search", "n_trials": n_trials,
        })
        return best_params

    def _evaluate_params(self, field: Any, params: Dict[str, float]) -> float:
        score = 0.0
        n_nodes = len(field.nodes)
        if n_nodes < 2:
            return 0.5
        positions = np.array([n.latent_pos for n in field.nodes.values()])
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        valid_dists = dists[dists < np.inf]
        if len(valid_dists) > 0:
            mean_dist = np.mean(valid_dists)
            std_dist = np.std(valid_dists)
            cv = std_dist / (mean_dist + 1e-8)
            score += float(max(0, 1.0 - cv)) * 0.4  # type: ignore[arg-type]
        phases = np.array([n.phase for n in field.nodes.values()])
        phase_order = np.abs(np.mean(np.exp(1j * phases)))
        score += phase_order * 0.3
        amplitudes = np.array([n.amplitude for n in field.nodes.values()])
        alive_ratio = np.mean(amplitudes > field.cfg.min_amplitude)
        score += alive_ratio * 0.3
        if "decay_rate" in params:
            decay_penalty = abs(
                params["decay_rate"] - field.cfg.decay_rate) * 10
            score -= decay_penalty * 0.1
        if field.stats.get("avg_response", 0) > 0:
            score += min(0.5, field.stats["avg_response"] * 0.5)
        return max(0.0, min(1.0, score))

    def apply_params(self, field: Any, params: Dict[str, float]):
        if "decay_rate" in params:
            field.cfg.decay_rate = params["decay_rate"]
            if field.learnable_kernel:
                field.learnable_kernel.decay_rate = params["decay_rate"]
        if "tension_threshold" in params:
            field.cfg.tension_threshold = params["tension_threshold"]
        if "phase_coupling" in params:
            field.cfg.phase_coupling = params["phase_coupling"]
            if field.meta_kernel:
                field.meta_kernel.base_phase_coupling = params["phase_coupling"]
        if "bandwidth" in params:
            field.cfg.bandwidth = params["bandwidth"]
            if field.meta_kernel:
                field.meta_kernel.base_bandwidth = params["bandwidth"]

    def should_optimize(self) -> bool:
        return self._step_counter > 0 and self._step_counter % self.optimization_freq == 0

    def get_state(self) -> Dict:
        return {
            "best_params": self._best_params,
            "optimization_history": self._optimization_history,
            "total_optimizations": self._total_optimizations,
            "optuna_available": self._optuna_available,
            "step_counter": self._step_counter,
            "last_optimization_time": self._last_optimization_time,
        }

    def load_state(self, state: Dict):
        self._best_params = state.get("best_params", {})
        self._optimization_history = state.get("optimization_history", [])
        self._total_optimizations = state.get("total_optimizations", 0)
        self._step_counter = state.get("step_counter", 0)
        self._last_optimization_time = state.get("last_optimization_time", 0.0)
