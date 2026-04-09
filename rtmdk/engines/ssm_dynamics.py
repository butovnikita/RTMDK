"""
rtmdk/engines/ssm_dynamics.py — State Space Model Dynamics (Mamba-inspired).

Replaces NeuralODE (O(N³)) with discrete State Space Models (O(N)).

SSM formulation:
  h_{t+1} = A · h_t + B · u_t    (state update)
  y_t     = C · h_t + D · u_t    (output)

Where:
  h_t: hidden state (N × state_dim)
  u_t: input (N × input_dim)
  A, B, C, D: learned parameter matrices

Discretization: Backward Euler for stability.
Reference: Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling"
"""

import numpy as np
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class SSMDynamics:
    """State Space Model for RTMDK field evolution.
    
    Replaces NeuralODE for O(N) field dynamics.
    
    Usage:
        ssm = SSMDynamics(state_dim=64, n_nodes=1000)
        h_next = ssm.step(h_current, u_input)
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        input_dim: int = 256,     # Matches latent_dim
        output_dim: int = 256,
        n_nodes: int = 1000,
        dt: float = 0.1,          # Discretization step
        learnable: bool = False,   # Whether parameters are trained
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_nodes = n_nodes
        self.dt = dt
        self.learnable = learnable
        
        # Initialize SSM parameters
        # A: stable dynamics (eigenvalues < 1)
        self.A = self._init_stable_matrix(state_dim)
        # B: input projection
        self.B = np.random.randn(state_dim, input_dim).astype(np.float32) * 0.01
        # C: output projection
        self.C = np.random.randn(output_dim, state_dim).astype(np.float32) * 0.01
        # D: skip connection
        self.D = np.eye(output_dim, input_dim).astype(np.float32) * 0.1
        
        # HiPPO initialization for long-range dependencies (optional)
        # self.A = self._init_hippo_matrix(state_dim)
        
        # Discretized parameters (for efficient inference)
        self._discretize()
        
        self.stats = {
            "steps": 0,
            "total_time_s": 0.0,
        }
    
    def _init_stable_matrix(self, dim: int) -> np.ndarray:
        """Initialize a stable A matrix (eigenvalues < 1)."""
        # Random orthogonal matrix scaled by 0.9
        Q, _ = np.linalg.qr(np.random.randn(dim, dim).astype(np.float32))
        eigenvalues = np.linspace(0.5, 0.95, dim).astype(np.float32)
        return Q @ np.diag(eigenvalues) @ Q.T
    
    def _init_hippo_matrix(self, dim: int) -> np.ndarray:
        """Initialize HiPPO matrix for long-range memory (simplified LegS)."""
        n = dim
        A = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i > j:
                    A[i, j] = (2 * i + 1) ** 0.5 * (2 * j + 1) ** 0.5
                elif i == j:
                    A[i, j] = -(i + 1)
        return A
    
    def _discretize(self):
        """Discretize continuous SSM using backward Euler.
        
        A_bar = (I - dt·A)^{-1}
        B_bar = (I - dt·A)^{-1} · dt·B
        """
        import time
        t0 = time.time()
        
        try:
            I = np.eye(self.state_dim, dtype=np.float32)
            A_inv = np.linalg.inv(I - self.dt * self.A)
            self.A_bar = A_inv
            self.B_bar = A_inv @ (self.dt * self.B)
        except np.linalg.LinAlgError:
            # Fallback: simple Euler
            logger.warning("SSM: Matrix inversion failed, using simple Euler")
            self.A_bar = np.eye(self.state_dim, dtype=np.float32) + self.dt * self.A
            self.B_bar = self.dt * self.B
        
        self.stats["discretize_time_s"] = time.time() - t0
    
    def step(
        self,
        h: np.ndarray,
        u: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One SSM step: h_{t+1} = A_bar·h_t + B_bar·u_t, y_t = C·h_t + D·u_t.
        
        Args:
            h: Hidden state (batch × state_dim) or (state_dim,)
            u: Input (batch × input_dim) or (input_dim,)
        
        Returns:
            (h_next, y): Next state and output
        """
        import time
        t0 = time.time()
        
        # Handle single input
        if h.ndim == 1:
            h = h.reshape(1, -1)
        if u.ndim == 1:
            u = u.reshape(1, -1)
        
        # State update: h_{t+1} = A_bar · h_t + B_bar · u_t
        h_next = h @ self.A_bar.T + u @ self.B_bar.T
        
        # Output: y_t = C · h_t + D · u_t
        y = h @ self.C.T + u @ self.D.T
        
        self.stats["steps"] += 1
        self.stats["total_time_s"] += time.time() - t0
        
        return h_next.squeeze(), y.squeeze()
    
    def update_params(self, A=None, B=None, C=None, D=None):
        """Update SSM parameters (for learning)."""
        if A is not None:
            self.A = A
        if B is not None:
            self.B = B
        if C is not None:
            self.C = C
        if D is not None:
            self.D = D
        self._discretize()
    
    def apply_to_field(self, memory_field):
        """Apply SSM dynamics to all nodes in the field.
        
        Each node's latent position is updated via SSM.
        """
        if not memory_field or not hasattr(memory_field, 'nodes'):
            return
        
        nodes = list(memory_field.nodes.values())
        if not nodes:
            return
        
        # Stack latent positions
        h = np.array([node.latent_pos for node in nodes])  # (N, latent_dim)
        
        # Input: node salience × amplitude (external stimulus)
        u = np.array([[node.salience * node.amplitude] * self.input_dim for node in nodes])
        
        # Apply SSM step
        h_next, _ = self.step(h, u)
        
        # Update nodes
        for i, node in enumerate(nodes):
            node.latent_pos = h_next[i]
    
    def get_stats(self) -> Dict:
        return {
            **self.stats,
            "state_dim": self.state_dim,
            "complexity": f"O(N·{self.state_dim}²)",
        }
