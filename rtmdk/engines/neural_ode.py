"""rtmdk/engines/neural_ode.py"""

from __future__ import annotations
from collections import deque
from typing import Any, Dict, List, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from scipy.integrate import odeint, solve_ivp

MemoryNode = Any


class NeuralODEDynamics:
    def __init__(
        self,
        latent_dim: int,
        noise_level: float = 0.01,
        time_horizon: float = 1.0,
        n_steps: int = 20,
        chunk_size: int = 256,
        solver: str = "RK45",
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ):
        self.latent_dim = latent_dim
        self.noise_level = noise_level
        self.time_horizon = time_horizon
        self.n_steps = n_steps
        self.chunk_size = chunk_size
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.alpha = 0.1
        self.beta = 0.05
        self.gamma = 0.02
        self.W = np.random.randn(latent_dim, latent_dim).astype(np.float32) * 0.01
        self._response_history: deque = deque(maxlen=100)
        self._state_history: List[NDArray] = []

    def _sigma(self, x: NDArray) -> NDArray:
        return np.tanh(x)

    def _dynamics(
        self,
        t: float,
        state: NDArray,
        input_signal: Optional[NDArray] = None,
        topology_gradient: Optional[NDArray] = None,
    ) -> NDArray:
        n_nodes = len(state) // self.latent_dim
        if n_nodes == 0:
            return state
        X = state.reshape(n_nodes, self.latent_dim)
        damping = -self.alpha * X
        nonlinear = self.W @ self._sigma(X.T)
        nonlinear = nonlinear.T
        if input_signal is not None:
            u = input_signal.reshape(n_nodes, self.latent_dim)
            attraction = self.beta * (u - X)
        else:
            attraction = 0.0
        if topology_gradient is not None:
            topo = self.gamma * topology_gradient.reshape(n_nodes, self.latent_dim)
        else:
            topo = 0.0
        dX = damping + nonlinear + attraction + topo
        return dX.flatten()

    def evolve(
        self,
        initial_state: NDArray,
        input_signal: Optional[NDArray] = None,
        topology_gradient: Optional[NDArray] = None,
        t_span: Optional[NDArray] = None,
    ) -> NDArray:
        if t_span is None:
            t_span = np.linspace(0, self.time_horizon, self.n_steps)
        n_nodes = len(initial_state) // self.latent_dim
        if n_nodes > self.chunk_size:
            return self._evolve_chunked(initial_state, input_signal, topology_gradient, t_span)

        def ode_func(t, state):
            return self._dynamics(t, state, input_signal, topology_gradient)

        solution = solve_ivp(
            ode_func,
            [t_span[0], t_span[-1]],
            initial_state.flatten(),
            t_eval=t_span,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
        )
        if solution.success:
            trajectory = solution.y.T
        else:
            trajectory = odeint(ode_func, initial_state.flatten(), t_span, atol=self.atol * 10, rtol=self.rtol * 10)
        self._state_history.append(trajectory[-1].copy())
        return trajectory

    def _evolve_chunked(
        self,
        initial_state: NDArray,
        input_signal: Optional[NDArray],
        topology_gradient: Optional[NDArray],
        t_span: NDArray,
    ) -> NDArray:
        n_nodes = len(initial_state) // self.latent_dim
        chunks = []
        for i in range(0, n_nodes, self.chunk_size):
            end = min(i + self.chunk_size, n_nodes)
            chunk_state = initial_state[i * self.latent_dim : end * self.latent_dim]
            chunk_input = (
                input_signal[i * self.latent_dim : end * self.latent_dim] if input_signal is not None else None
            )
            chunk_topo = (
                topology_gradient[i * self.latent_dim : end * self.latent_dim]
                if topology_gradient is not None
                else None
            )

            def ode_func(t, state, ci=chunk_input, ct=chunk_topo):
                return self._dynamics(t, state, ci, ct)

            sol = solve_ivp(
                ode_func,
                [t_span[0], t_span[-1]],
                chunk_state.flatten(),
                t_eval=t_span,
                method=self.solver,
                atol=self.atol,
                rtol=self.rtol,
            )
            if sol.success:
                chunks.append(sol.y.T)
            else:
                chunks.append(odeint(ode_func, chunk_state.flatten(), t_span, atol=self.atol, rtol=self.rtol))
        return np.concatenate(chunks, axis=1)

    def evolve_with_noise(
        self,
        initial_state: NDArray,
        input_signal: Optional[NDArray] = None,
        topology_gradient: Optional[NDArray] = None,
        dt: float = 0.05,
    ) -> NDArray:
        n_steps = int(self.time_horizon / dt)
        state = initial_state.flatten().copy()
        trajectory = [state.copy()]
        for _ in range(n_steps):
            deterministic = self._dynamics(0, state, input_signal, topology_gradient) * dt
            noise = self.noise_level * np.random.randn(len(state)) * np.sqrt(dt)
            state = state + deterministic + noise
            trajectory.append(state.copy())
        self._state_history.append(trajectory[-1].copy())
        return np.array(trajectory)

    def compute_topology_gradient(self, nodes: Dict[str, Any]) -> Optional[NDArray]:
        if len(nodes) < 2:
            return None
        node_ids = list(nodes.keys())
        positions = np.array([nodes[nid].latent_pos for nid in node_ids])
        n = len(positions)
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        gradient = np.zeros_like(positions)
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < 2.0:
                    direction = (positions[i] - positions[j]) / (dists[i, j] + 1e-8)
                    gradient[i] += direction * 0.01
                    gradient[j] -= direction * 0.01
        return gradient.flatten()

    def compute_response_smoothness(self) -> float:
        if len(self._response_history) < 2:
            return 1.0
        responses = np.array(self._response_history)
        std = np.std(responses)
        return float(max(0.0, 1.0 - std))  # type: ignore[arg-type]

    def record_response(self, response: float):
        self._response_history.append(response)

    def get_state(self) -> Dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "W": self.W.tolist(),
            "noise_level": self.noise_level,
            "smoothness": self.compute_response_smoothness(),
        }

    def load_state(self, state: Dict):
        self.alpha = state.get("alpha", self.alpha)
        self.beta = state.get("beta", self.beta)
        self.gamma = state.get("gamma", self.gamma)
        if "W" in state:
            self.W = np.array(state["W"], dtype=np.float32)
        self.noise_level = state.get("noise_level", self.noise_level)
