"""Tests for rtmdk/memory/cpen_ode.py — parent-child coupled ODE dynamics."""

from types import SimpleNamespace

import numpy as np
import pytest

from rtmdk.memory.cpen_ode import CPENChildODE, CPENODECoupledSystem, CPENParentODE

LATENT = 4
NODES = 3


@pytest.fixture(autouse=True)
def fixed_seed():
    np.random.seed(0)


@pytest.fixture
def parent():
    return CPENParentODE(latent_dim=LATENT, num_nodes=NODES)


@pytest.fixture
def child():
    return CPENChildODE(latent_dim=LATENT, num_nodes=NODES, hebbian_eta=0.01, decay=0.01)


class TestParentODE:
    def test_init_dims(self, parent):
        assert parent.state_dim == LATENT * NODES
        assert parent.W.shape == (LATENT * NODES, LATENT * NODES)
        # Small-weight init: spectral entries stay well below 1
        assert np.abs(parent.W).max() < 1.0

    def test_set_input_exact_size(self, parent):
        v = np.arange(parent.state_dim, dtype=float)
        parent.set_input(v)
        np.testing.assert_array_equal(parent.input, v)

    def test_set_input_latent_size_is_tiled(self, parent):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        parent.set_input(v)
        np.testing.assert_array_equal(parent.input, np.tile(v, NODES))

    def test_set_input_odd_size_is_padded(self, parent):
        v = np.ones(7)
        parent.set_input(v)
        assert parent.input.shape == (parent.state_dim,)
        np.testing.assert_array_equal(parent.input[:7], v)
        np.testing.assert_array_equal(parent.input[7:], np.zeros(parent.state_dim - 7))

    def test_set_input_oversize_is_truncated(self, parent):
        v = np.arange(100, dtype=float)
        parent.set_input(v)
        np.testing.assert_array_equal(parent.input, v[: parent.state_dim])

    def test_dynamics_zero_state_zero_input(self, parent):
        parent.set_input(np.zeros(parent.state_dim))
        dx = parent.dynamics(0.0, np.zeros(parent.state_dim))
        np.testing.assert_array_equal(dx, np.zeros(parent.state_dim))

    def test_dynamics_at_origin_equals_input(self, parent):
        v = np.random.randn(parent.state_dim)
        parent.set_input(v)
        dx = parent.dynamics(0.0, np.zeros(parent.state_dim))
        np.testing.assert_allclose(dx, v)

    def test_dynamics_matches_formula(self, parent):
        v = np.random.randn(parent.state_dim)
        parent.set_input(v)
        x = np.random.randn(parent.state_dim)
        dx = parent.dynamics(0.0, x)
        np.testing.assert_allclose(dx, -x + parent.W @ np.tanh(x) + v)


class TestChildODE:
    def test_init_dims(self, child):
        assert child.state_dim == 2 * NODES

    def test_set_input_latent_major(self, child):
        m = np.random.randn(LATENT, NODES)
        child.set_input(m)
        np.testing.assert_array_equal(child.inputs, m)

    def test_set_input_node_major_is_transposed(self, child):
        m = np.random.randn(NODES, LATENT)
        child.set_input(m)
        np.testing.assert_array_equal(child.inputs, m.T)

    def test_set_input_flat_is_reshaped(self, child):
        m = np.random.randn(LATENT * NODES)
        child.set_input(m)
        np.testing.assert_array_equal(child.inputs, m.reshape(LATENT, NODES))

    def test_set_input_invalid_gives_zeros(self, child):
        child.set_input(np.zeros((5, 5)))
        np.testing.assert_array_equal(child.inputs, np.zeros((LATENT, NODES)))

    def test_dynamics_shape(self, child):
        child.set_input(np.random.randn(LATENT, NODES))
        state = np.concatenate([np.ones(NODES), np.zeros(NODES)])
        dx = child.dynamics(0.0, state)
        assert dx.shape == (2 * NODES,)

    def test_amplitude_hebbian_rule(self, child):
        m = np.random.randn(LATENT, NODES)
        child.set_input(m)
        a = np.array([1.0, 2.0, 0.5])
        state = np.concatenate([a, np.zeros(NODES)])

        dx = child.dynamics(0.0, state)
        expected = child.hebbian_eta * (np.linalg.norm(m, axis=0) * a - child.decay * a)
        np.testing.assert_allclose(dx[:NODES], expected)

    def test_equal_phases_have_no_coupling(self, child):
        """Identical phases → sin(phi_j - phi_i) = 0 → dphi/dt = natural freq."""
        child.set_input(np.zeros((LATENT, NODES)))
        phi = np.full(NODES, 0.7)
        state = np.concatenate([np.ones(NODES), phi])

        dx = child.dynamics(0.0, state)
        np.testing.assert_allclose(dx[NODES:], np.full(NODES, 0.1), atol=1e-12)

    def test_phase_coupling_formula(self, child):
        child.set_input(np.zeros((LATENT, NODES)))
        phi = np.array([0.0, np.pi / 2, np.pi])
        state = np.concatenate([np.ones(NODES), phi])

        dx = child.dynamics(0.0, state)
        sum_sin = np.sin(phi).sum()
        sum_cos = np.cos(phi).sum()
        expected = 0.1 + 0.1 * (sum_sin * np.cos(phi) - sum_cos * np.sin(phi))
        np.testing.assert_allclose(dx[NODES:], expected)


class TestCoupledSystem:
    @pytest.fixture
    def system(self):
        return CPENODECoupledSystem(latent_dim=LATENT, num_nodes=NODES, input_dim=LATENT)

    def test_state_dim(self, system):
        assert system.state_dim == LATENT * NODES + 2 * NODES

    def test_set_input_reaches_both_subsystems(self, system):
        v = np.array([1.0, -1.0, 0.5, 2.0])
        system.set_input(v)

        np.testing.assert_array_equal(system.parent_ode.input, np.tile(v, NODES))
        expected_child = np.tile(v.reshape(-1, 1), (1, NODES))
        np.testing.assert_array_equal(system.child_ode.inputs, expected_child)

    def test_dynamics_is_concat_of_subsystems(self, system):
        system.set_input(np.random.randn(LATENT))
        parent_state = np.random.randn(LATENT * NODES)
        child_state = np.concatenate([np.ones(NODES), np.zeros(NODES)])
        state = np.concatenate([parent_state, child_state])

        dx = system.dynamics(0.0, state)
        np.testing.assert_allclose(dx[: LATENT * NODES], system.parent_ode.dynamics(0.0, parent_state))
        np.testing.assert_allclose(dx[LATENT * NODES :], system.child_ode.dynamics(0.0, child_state))

    def test_integrate(self, system):
        system.set_input(np.random.randn(LATENT) * 0.1)
        initial = np.concatenate(
            [
                np.random.randn(LATENT * NODES) * 0.1,
                np.ones(NODES),
                np.zeros(NODES),
            ]
        )
        t_eval = np.linspace(0.0, 0.5, 6)
        sol = system.integrate((0.0, 0.5), initial, t_eval=t_eval)

        assert sol.success
        assert sol.y.shape == (system.state_dim, len(t_eval))
        assert np.all(np.isfinite(sol.y))
        # First column is the initial state
        np.testing.assert_allclose(sol.y[:, 0], initial, atol=1e-8)
        # Amplitudes start at 1 and evolve slowly (eta=0.01) → stay near 1
        amps = sol.y[LATENT * NODES : LATENT * NODES + NODES, -1]
        np.testing.assert_allclose(amps, np.ones(NODES), atol=0.1)

    def test_integrate_failure_raises(self, system, monkeypatch):
        failed = SimpleNamespace(success=False, message="boom")
        monkeypatch.setattr("rtmdk.memory.cpen_ode.solve_ivp", lambda *a, **k: failed)

        with pytest.raises(RuntimeError, match="ODE integration failed: boom"):
            system.integrate((0.0, 1.0), np.zeros(system.state_dim))
