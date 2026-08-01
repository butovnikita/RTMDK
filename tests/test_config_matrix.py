"""Config validation matrix — ensures key parameter combinations initialize."""

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKField


class TestConfigMatrix:
    def _check(self, **kwargs):
        cfg = RTMDKConfig(**kwargs)
        field = RTMDKField(cfg)
        assert field.cfg is not None

    def test_default_config(self):
        self._check()

    def test_high_dim(self):
        self._check(latent_dim=256)

    def test_low_dim(self):
        self._check(latent_dim=16)

    def test_sot_byte_default(self):
        self._check(sot_enabled=True)

    def test_sot_word_mode(self):
        self._check(sot_enabled=True, sot_tokenization_mode="word")

    def test_sot_word_with_cooccurrence_limit(self):
        self._check(sot_enabled=True, sot_tokenization_mode="word", sot_max_cooccurrence=100)

    def test_sot_byte_subword_seed(self):
        self._check(sot_enabled=True, sot_subword_seed=True)

    def test_sot_byte_attention(self):
        self._check(sot_enabled=True, sot_attention_pooling=True)

    def test_sot_byte_skipgram(self):
        self._check(sot_enabled=True, sot_skipgram_window=5)

    def test_sot_byte_hard_negatives(self):
        self._check(sot_enabled=True, sot_hard_negatives=True)

    def test_sot_byte_retrieval_feedback(self):
        self._check(sot_enabled=True, sot_retrieval_feedback=True)

    def test_hyperbolic(self):
        self._check(hyperbolic=True)

    def test_conformal_prediction(self):
        self._check(conformal_prediction=True)

    def test_meta_adaptive(self):
        self._check(meta_adaptive=True)

    def test_spectral_consolidation(self):
        self._check(spectral_consolidation=True)

    def test_kalman_filter(self):
        self._check(enable_kalman_filter=True)

    def test_engrams(self):
        self._check(enable_engrams=True)

    def test_security(self):
        self._check(security_enabled=True)

    def test_version_control(self):
        self._check(version_control=True)

    def test_full_stress(self):
        self._check(
            latent_dim=64,
            sot_enabled=True,
            sot_tokenization_mode="word",
            sot_subword_seed=True,
            sot_attention_pooling=True,
            sot_hard_negatives=True,
            sot_retrieval_feedback=True,
            sot_skipgram_window=3,
            sot_max_cooccurrence=50,
            hyperbolic=True,
            conformal_prediction=True,
            meta_adaptive=True,
            spectral_consolidation=True,
            enable_kalman_filter=True,
            enable_engrams=True,
            security_enabled=True,
        )
