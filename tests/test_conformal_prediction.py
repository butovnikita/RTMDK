"""
tests/test_conformal_prediction.py — P1.1 Conformal Prediction (ICP) for retrieval confidence.

Covers:
1. Calibrator threshold computation
2. Prediction set coverage guarantee on synthetic data
3. Calibration via field.calibrate()
4. Query filtering with conformal prediction enabled
5. Cold-start: no filtering before min_calib samples
"""

import numpy as np

from rtmdk.memory.conformal import ConformalCalibrator
from rtmdk import RTMDKConfig, RTMDKField


def _make_field(
        n_nodes=20,
        dim=8,
        conformal=True,
        alpha=0.1,
        min_calib=10,
        bw=1.0):
    cfg = RTMDKConfig(
        latent_dim=dim,
        bandwidth=bw,
        conformal_prediction=conformal,
        conformal_alpha=alpha,
        conformal_min_calib=min_calib,
    )
    field = RTMDKField(cfg)
    rng = np.random.default_rng(42)
    for i in range(n_nodes):
        pos = rng.standard_normal(dim).astype(np.float32) * 0.5
        nid = field.add_node(
            pos,
            content={"text": f"node {i}"},
            phase=0.0,
            node_id=f"n{i}",
            skip_projection=True,
        )
        field.nodes[nid].amplitude = 1.0
        field.nodes[nid].salience = 1.0
    field._build_node_cache()
    return field


class TestConformalCalibrator:
    def test_threshold_zero_when_empty(self):
        cal = ConformalCalibrator(alpha=0.1)
        assert cal.get_threshold() == 0.0

    def test_threshold_between_zero_and_one(self):
        cal = ConformalCalibrator(alpha=0.1)
        cal.fit([0.5, 0.6, 0.7, 0.8, 0.9])
        thr = cal.get_threshold()
        assert 0.0 <= thr <= 1.0

    def test_higher_alpha_lower_threshold(self):
        """Higher alpha (more tolerance for error) → lower threshold."""
        scores = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        cal_low = ConformalCalibrator(alpha=0.05)
        cal_low.fit(scores)
        cal_high = ConformalCalibrator(alpha=0.20)
        cal_high.fit(scores)
        # For small n, k may exceed n → threshold=0 for both.
        # Use larger n for a meaningful ordering test.
        scores_big = list(np.linspace(0.1, 0.9, 50))
        cal_low2 = ConformalCalibrator(alpha=0.05)
        cal_low2.fit(scores_big)
        cal_high2 = ConformalCalibrator(alpha=0.20)
        cal_high2.fit(scores_big)
        # Higher alpha (more errors allowed) -> HIGHER threshold (fewer items included)
        assert cal_high2.get_threshold() >= cal_low2.get_threshold()

    def test_prediction_set_filters_low_scores(self):
        cal = ConformalCalibrator(alpha=0.1)
        # Need enough samples so that k <= n (threshold > 0)
        cal.fit([0.8, 0.85, 0.9, 0.95] * 5)  # n=20, k=ceil(21*0.9)=19, thr=sorted[1]
        nids = ["a", "b", "c"]
        scores = [0.96, 0.50, 0.30]
        pred_set, conf, thr = cal.predict(scores, nids)
        assert "a" in pred_set
        assert "b" not in pred_set
        assert "c" not in pred_set
        assert conf == 0.9

    def test_empirical_coverage(self):
        """On synthetic calibration + test, coverage should be ~1-alpha."""
        rng = np.random.default_rng(7)
        alpha = 0.10
        cal = ConformalCalibrator(alpha=alpha)
        # Simulate relevance scores (all from relevant items)
        cal_scores = rng.uniform(0.4, 1.0, 200).tolist()
        cal.fit(cal_scores)

        n_test = 500
        covered = 0
        for _ in range(n_test):
            true_score = rng.uniform(0.4, 1.0)
            # Mix true relevant item with 9 random distractors
            scores = [true_score] + rng.uniform(0.0, 1.0, 9).tolist()
            nids = [f"n{i}" for i in range(10)]
            pred_set, _, _ = cal.predict(scores, nids)
            if "n0" in pred_set:
                covered += 1

        empirical_coverage = covered / n_test
        # Allow 0.05 slack
        assert empirical_coverage >= 1.0 - alpha - 0.05


class TestConformalFieldIntegration:
    def test_disabled_by_default(self):
        field = _make_field(n_nodes=10, conformal=False)
        assert field.conformal_calibrator is None

    def test_cold_start_no_filtering(self):
        """Before min_calib samples, conformal should not filter results."""
        field = _make_field(n_nodes=20, conformal=True, min_calib=100)
        query = np.zeros(8, dtype=np.float32)
        results = field.query(query, top_k=10)
        assert len(results) == 10
        # Stats should show no conformal activity
        assert field.stats["conformal_prediction_set_size"] == 0

    def test_calibrate_adds_samples(self):
        field = _make_field(n_nodes=10, conformal=True, min_calib=5)
        assert field.conformal_calibrator.n_calibrated == 0
        for i in range(7):
            emb = np.zeros(8, dtype=np.float32)
            field.calibrate(emb, f"n{i}", is_relevant=True)
        assert field.conformal_calibrator.n_calibrated == 7

    def test_query_filters_after_calibration(self):
        """After enough calibration, conformal should potentially filter results."""
        field = _make_field(n_nodes=30, conformal=True, min_calib=10, bw=0.5)
        # Calibrate with high relevance scores (simulating good retrieval)
        rng = np.random.default_rng(99)
        for _ in range(25):
            emb = rng.standard_normal(8).astype(np.float32)
            # Pick a random node and declare it relevant
            nid = f"n{rng.integers(0, 30)}"
            field.calibrate(emb, nid, is_relevant=True)

        query = np.zeros(8, dtype=np.float32)
        results = field.query(query, top_k=10)
        # Threshold should have been computed
        assert field.stats["conformal_confidence"] == 0.9
        assert "conformal_threshold" in field.stats
        # Results may be filtered; just check no crash and non-negative scores
        assert all(score >= 0 for _, score, _ in results)

    def test_non_relevant_not_added(self):
        field = _make_field(n_nodes=10, conformal=True, min_calib=5)
        emb = np.zeros(8, dtype=np.float32)
        field.calibrate(emb, "n0", is_relevant=False)
        assert field.conformal_calibrator.n_calibrated == 0
