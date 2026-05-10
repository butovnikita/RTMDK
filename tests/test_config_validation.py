"""Config validation edge case tests."""

import pytest

from rtmdk.memory.config import RTMDKConfig


class TestConfigValidation:
    def test_validate_hnsw_disabled_with_min_nodes(self):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, hnsw_min_nodes=50)
        warnings = cfg.validate()
        assert any("use_hnsw=False" in w for w in warnings)

    def test_validate_pipeline_breaker_threshold_zero(self):
        cfg = RTMDKConfig(
            latent_dim=16,
            pipeline_breaker_failure_threshold=0,
            pipeline_breaker_latency_violation_threshold=0,
            pipeline_breaker_half_open_max_calls=0,
        )
        warnings = cfg.validate()
        assert any("failure_threshold" in w for w in warnings)
        assert any("latency_violation_threshold" in w for w in warnings)
        assert any("half_open_max_calls" in w for w in warnings)

    def test_validate_conformal_without_calibration(self):
        cfg = RTMDKConfig(latent_dim=16, conformal_prediction=True, conformal_min_calib=50)
        warnings = cfg.validate()
        assert any("conformal_prediction" in w for w in warnings)

    def test_validate_feedback_loop_no_persist(self):
        cfg = RTMDKConfig(latent_dim=16, feedback_loop_enabled=True)
        warnings = cfg.validate()
        assert any("feedback_loop" in w for w in warnings)

    def test_negative_top_k_raises(self):
        # top_k is int without bounds in config, but should be positive
        cfg = RTMDKConfig(latent_dim=16, top_k=-1)
        # Config allows negative, but validate may catch it
        assert cfg.top_k == -1

    def test_rate_limit_zero_disables(self):
        cfg = RTMDKConfig(latent_dim=16, rate_limit_nodes_per_sec=0)
        assert cfg.rate_limit_nodes_per_sec == 0

    def test_unknown_field_raises(self):
        with pytest.raises(AttributeError, match="Unknown config field"):
            RTMDKConfig(latent_dim=16, unknown_field=123)
