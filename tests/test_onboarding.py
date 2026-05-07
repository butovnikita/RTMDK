"""Tests for rtmdk.production.onboarding."""

from rtmdk.production.onboarding import OnboardingWizard


class TestOnboardingWizard:
    def test_get_steps(self):
        wizard = OnboardingWizard()
        steps = wizard.get_steps()
        assert len(steps) == 4
        assert steps[0]["step"] == 1

    def test_complete_step(self):
        wizard = OnboardingWizard()
        wizard.complete_step(1, {"option": "personal"})
        assert 1 in wizard._completed_steps

    def test_get_progress(self):
        wizard = OnboardingWizard()
        progress = wizard.get_progress()
        assert progress["total_steps"] == 4
        assert progress["completed_steps"] == 0
        assert progress["progress_percent"] == 0.0

    def test_get_progress_partial(self):
        wizard = OnboardingWizard()
        wizard.complete_step(1, {})
        progress = wizard.get_progress()
        assert progress["completed_steps"] == 1
        assert progress["next_step"] == 2

    def test_get_recommended_config(self):
        wizard = OnboardingWizard()
        wizard.complete_step(1, {"option": "production"})
        cfg = wizard.get_recommended_config()
        assert cfg["preset"] == "production"
        assert len(cfg["next_steps"]) == 4
