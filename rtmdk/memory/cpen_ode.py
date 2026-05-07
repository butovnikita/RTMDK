"""
CPEN+ ODE Dynamics for Resonance-Topological Memory

Implements parent-child ODE coupling:
- Parent ODE: Governs slow, causal latent dynamics (embedding space)
- Child ODE (per node): Governs fast, local amplitude/phase updates via Hebbian-like rule
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, Tuple


class CPENParentODE:
    """
    Parent ODE governing slow causal latent dynamics.

    Dynamics: dx/dt = -x + W * tanh(x) + I
    where x is the concatenated latent positions of all nodes,
    W is a weight matrix, I is the input.
    """

    def __init__(self, latent_dim: int, num_nodes: int, input_dim: int = 768):
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.state_dim = latent_dim * num_nodes
        # Initialize weight matrix with small random values
        self.W = np.random.randn(self.state_dim, self.state_dim) * 0.1

    def set_input(self, input_vector: np.ndarray):
        """Set external input vector (should match state_dim)."""
        if input_vector.shape[0] != self.state_dim:
            # If input is different size, project or tile
            if input_vector.shape[0] == self.latent_dim:
                # Tile the input to all nodes
                self.input = np.tile(input_vector, self.num_nodes)
            else:
                # Pad or truncate
                self.input = np.zeros(self.state_dim)
                self.input[:min(self.state_dim, input_vector.shape[0])] = input_vector[:min(
                    self.state_dim, input_vector.shape[0])]
        else:
            self.input = input_vector

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute parent ODE dynamics."""
        # Parent dynamics: dx/dt = -x + W * tanh(x) + I
        x = state
        dxdt = -x + self.W @ np.tanh(x) + self.input
        return dxdt


class CPENChildODE:
    """
    Child ODE per node governing fast local amplitude/phase updates.

    For each node i:
    dA_i/dt = eta * (input_i * A_i - decay * A_i)
    dphi_i/dt = omega_i + coupling * sum_j sin(phi_j - phi_i)
    where A_i is amplitude, phi_i is phase.
    """

    def __init__(
            self,
            latent_dim: int,
            num_nodes: int,
            hebbian_eta: float = 0.01,
            decay: float = 0.01):
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.hebbian_eta = hebbian_eta
        self.decay = decay
        # [A_0, ..., A_{N-1}, phi_0, ..., phi_{N-1}]
        self.state_dim = 2 * num_nodes

    def set_input(self, input_vectors: np.ndarray):
        """Set input for each node (latent_dim x num_nodes)."""
        # input_vectors shape: (latent_dim, num_nodes) or (num_nodes, latent_dim)
        # We'll assume (latent_dim, num_nodes) for convenience
        if input_vectors.shape == (self.latent_dim, self.num_nodes):
            self.inputs = input_vectors
        elif input_vectors.shape == (self.num_nodes, self.latent_dim):
            self.inputs = input_vectors.T
        else:
            # Assume flat array of length latent_dim * num_nodes
            flat = input_vectors.flatten()
            if flat.size == self.latent_dim * self.num_nodes:
                self.inputs = flat.reshape(self.latent_dim, self.num_nodes)
            else:
                # Default to zero
                self.inputs = np.zeros((self.latent_dim, self.num_nodes))

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute child ODE dynamics for all nodes."""
        A = state[:self.num_nodes]          # amplitudes
        phi = state[self.num_nodes:]        # phases

        # Hebbian update for amplitude: dA/dt = eta * (input * A - decay * A)
        # input per node: we take the norm of the input vector for that node
        input_norms = np.linalg.norm(self.inputs, axis=0)  # shape (num_nodes,)
        dA_dt = self.hebbian_eta * (input_norms * A - self.decay * A)

        # Phase dynamics: dphi/dt = natural_freq + coupling * sum_j sin(phi_j - phi_i)
        # For simplicity, set natural_freq = 0.1 and coupling = 0.1
        natural_freq = 0.1 * np.ones_like(phi)
        coupling = 0.1
        # Compute coupling term using vectorized version
        # For each i, sum_j sin(phi_j - phi_i) = sum_j sin(phi_j) * cos(phi_i)
        # - cos(phi_j) * sin(phi_i)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sum_sin_phi = np.sum(sin_phi)
        sum_cos_phi = np.sum(cos_phi)
        dphi_dt = natural_freq + coupling * \
            (sum_sin_phi * cos_phi - sum_cos_phi * sin_phi)

        return np.concatenate([dA_dt, dphi_dt])


class CPENODECoupledSystem:
    """
    Combined parent-child ODE system.

    State vector: [parent_state (latent_dim * num_nodes), child_state (2 * num_nodes)]
    """

    def __init__(self, latent_dim: int, num_nodes: int, input_dim: int = 768,
                 hebbian_eta: float = 0.01, decay: float = 0.01):
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.parent_ode = CPENParentODE(latent_dim, num_nodes, input_dim)
        self.child_ode = CPENChildODE(
            latent_dim, num_nodes, hebbian_eta, decay)
        self.state_dim = latent_dim * num_nodes + 2 * num_nodes

    def set_input(self, input_vector: np.ndarray):
        """Set input for the system."""
        # Input is assumed to be of shape (latent_dim,) for a single token embedding
        # We need to provide it to parent and child.
        # For parent: tile or project to parent state dim
        # For child: treat as input for each node (same for all nodes)
        self.parent_ode.set_input(input_vector)
        # For child, we need input per node: we'll use the same input vector for each node
        # Create a matrix of shape (latent_dim, num_nodes) where each column is
        # input_vector
        self.child_ode.set_input(
            np.tile(input_vector.reshape(-1, 1), (1, self.num_nodes)))

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute combined dynamics."""
        # Split state
        parent_size = self.latent_dim * self.num_nodes
        parent_state = state[:parent_size]
        child_state = state[parent_size:]

        # Temporarily set the states in the sub-ODEs
        # We'll compute dynamics by calling each sub-ODE's dynamics method
        # but we need to set their internal state.
        # Instead, we can compute directly:
        parent_dx = self.parent_ode.dynamics(t, parent_state)
        child_dx = self.child_ode.dynamics(t, child_state)

        return np.concatenate([parent_dx, child_dx])

    def integrate(self, t_span: Tuple[float, float], initial_state: np.ndarray,
                  t_eval: Optional[np.ndarray] = None, method: str = 'RK45',
                  atol: float = 1e-6, rtol: float = 1e-3):
        """Integrate the ODE system."""
        sol = solve_ivp(self.dynamics, t_span, initial_state, method=method,
                        t_eval=t_eval, atol=atol, rtol=rtol)
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        return sol


# Example usage (for testing)
if __name__ == "__main__":
    # Example: 5 nodes, latent dim 64
    latent_dim = 64
    num_nodes = 5
    input_dim = 768

    system = CPENODECoupledSystem(latent_dim, num_nodes, input_dim)

    # Initial state: random latent positions, amplitudes=1, phases=0
    initial_parent = np.random.randn(latent_dim * num_nodes) * 0.1
    initial_child = np.concatenate([np.ones(num_nodes), np.zeros(num_nodes)])
    initial_state = np.concatenate([initial_parent, initial_child])

    # Input: random embedding
    input_vector = np.random.randn(input_dim)
    system.set_input(input_vector)

    # Integrate from t=0 to t=1
    t_span = (0.0, 1.0)
    sol = system.integrate(
        t_span,
        initial_state,
        t_eval=np.linspace(
            0,
            1,
            100))

    print(f"Integration successful. Shape of solution: {sol.y.shape}")
