"""
Test hierarchical RTMDKConfig (P3 refactor).

Covers:
- Nested group construction
- Flat-field backward compatibility (__getattr__ / __setattr__)
- asdict() flat output
- __eq__ and __repr__
- __post_init__ derived defaults
- Env-var overrides
- Unknown field AttributeError
- Serialization round-trip preserves flat dict
"""

import pytest
import numpy as np
from rtmdk.memory.config import (
    RTMDKConfig,
    CoreConfig,
    RetrievalConfig,
    LearningConfig,
    DynamicsConfig,
    InferenceConfig,
    MemorySystemConfig,
    ProductionConfig,
    RoutingConfig,
    SOTConfig,
)
from rtmdk.memory.core import RTMDKField
from rtmdk.memory.serialization import FieldSerializer


class TestHierarchicalConfig:
    def test_default_construction(self):
        cfg = RTMDKConfig()
        assert isinstance(cfg.core, CoreConfig)
        assert isinstance(cfg.retrieval, RetrievalConfig)
        assert isinstance(cfg.learning, LearningConfig)
        assert isinstance(cfg.dynamics, DynamicsConfig)
        assert isinstance(cfg.inference, InferenceConfig)
        assert isinstance(cfg.memory, MemorySystemConfig)
        assert isinstance(cfg.production, ProductionConfig)
        assert isinstance(cfg.routing, RoutingConfig)
        assert isinstance(cfg.sot, SOTConfig)

    def test_flat_field_read(self):
        cfg = RTMDKConfig()
        assert cfg.latent_dim == 64
        assert cfg.bandwidth == 1.0
        assert cfg.top_k == 5
        assert cfg.phase_coupling == 0.3

    def test_flat_field_write(self):
        cfg = RTMDKConfig()
        cfg.latent_dim = 128
        assert cfg.core.latent_dim == 128
        assert cfg.latent_dim == 128

    def test_constructor_flat_kwargs(self):
        cfg = RTMDKConfig(latent_dim=256, bandwidth=2.0, top_k=10)
        assert cfg.latent_dim == 256
        assert cfg.bandwidth == 2.0
        assert cfg.top_k == 10
        assert cfg.core.latent_dim == 256

    def test_constructor_nested_kwargs(self):
        core = CoreConfig(latent_dim=512)
        cfg = RTMDKConfig(core=core)
        assert cfg.latent_dim == 512
        assert cfg.core.latent_dim == 512
        # Other groups still default
        assert cfg.bandwidth == 1.0

    def test_constructor_mixed_kwargs(self):
        core = CoreConfig(latent_dim=512)
        cfg = RTMDKConfig(core=core, bandwidth=3.0)
        assert cfg.latent_dim == 512
        assert cfg.bandwidth == 3.0

    def test_asdict_is_flat(self):
        cfg = RTMDKConfig(latent_dim=128)
        d = cfg.asdict()
        assert isinstance(d, dict)
        assert "latent_dim" in d
        assert "bandwidth" in d
        assert d["latent_dim"] == 128
        assert "core" not in d  # nested group names must not appear

    def test_eq(self):
        cfg1 = RTMDKConfig(latent_dim=128)
        cfg2 = RTMDKConfig(latent_dim=128)
        cfg3 = RTMDKConfig(latent_dim=256)
        assert cfg1 == cfg2
        assert cfg1 != cfg3
        assert cfg1 != "not a config"

    def test_repr(self):
        cfg = RTMDKConfig()
        r = repr(cfg)
        assert r.startswith("RTMDKConfig(")
        assert "core=" in r
        assert "retrieval=" in r

    def test_post_init_derived_defaults(self):
        cfg = RTMDKConfig()
        assert cfg.core.pca_n_components == cfg.core.latent_dim
        assert cfg.retrieval.modality_phase_shifts == {
            "text": 0.0,
            "audio": np.pi / 3,
            "image": np.pi / 2,
            "video": np.pi,
        }

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("RTMDK_LATENT_DIM", "99")
        monkeypatch.setenv("RTMDK_BANDWIDTH", "2.5")
        monkeypatch.setenv("RTMDK_USE_HNSW", "true")
        cfg = RTMDKConfig()
        assert cfg.latent_dim == 99
        assert cfg.bandwidth == 2.5
        assert cfg.use_hnsw is True

    def test_env_override_invalid_warns(self, monkeypatch, caplog):
        monkeypatch.setenv("RTMDK_LATENT_DIM", "not_an_int")
        import logging

        with caplog.at_level(logging.WARNING, logger="rtmdk"):
            cfg = RTMDKConfig()
        assert "Invalid env var RTMDK_LATENT_DIM" in caplog.text
        assert cfg.latent_dim == 64  # falls back to default

    def test_unknown_field_raises(self):
        cfg = RTMDKConfig()
        with pytest.raises(AttributeError):
            _ = cfg.nonexistent_field
        with pytest.raises(AttributeError):
            cfg.nonexistent_field = 1
        with pytest.raises(AttributeError):
            RTMDKConfig(nonexistent_field=1)

    def test_pipeline_breaker_validation(self):
        from rtmdk.memory.config import ProductionConfig

        prod = ProductionConfig(
            pipeline_breaker_failure_threshold=0,
            pipeline_breaker_latency_violation_threshold=0,
            pipeline_breaker_recovery_timeout_ms=500,
            pipeline_breaker_half_open_max_calls=0,
            pipeline_breaker_thresholds={"embed": -1.0},
        )
        cfg = RTMDKConfig(production=prod)
        warnings = cfg.validate()
        assert any("failure_threshold" in w for w in warnings)
        assert any("latency_violation_threshold" in w for w in warnings)
        assert any("recovery_timeout_ms" in w for w in warnings)
        assert any("half_open_max_calls" in w for w in warnings)
        assert any("embed" in w for w in warnings)

    def test_pipeline_breaker_valid_no_warnings(self):
        cfg = RTMDKConfig()
        warnings = cfg.validate()
        pipeline_warnings = [w for w in warnings if "pipeline_breaker" in w]
        assert len(pipeline_warnings) == 0

    def test_serialization_roundtrip(self):
        """FieldSerializer must produce a flat config dict."""
        cfg = RTMDKConfig(latent_dim=128, bandwidth=2.0)
        field = RTMDKField(config=cfg)
        data = FieldSerializer.field_to_dict(field)
        config_dict = data["config"]
        assert isinstance(config_dict, dict)
        assert "latent_dim" in config_dict
        assert config_dict["latent_dim"] == 128
        assert "bandwidth" in config_dict
        assert "core" not in config_dict
        # Verify enum serialization
        assert config_dict["consolidation_mode"] == "dialectical"
        assert config_dict["context_format"] == "plain"
