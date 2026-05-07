"""
rtmdk/production/onboarding.py — User Onboarding Wizard.

Step-by-step setup guide for new RTMDK users.
"""

from typing import Dict, List, Any


ONBOARDING_STEPS = [{"step": 1,
                     "title": "Choose Your Use Case",
                     "description": "Select how you'll use RTMDK",
                     "options": {"personal": {"preset": "local",
                                              "desc": "Personal assistant, <10K memories"},
                                 "production": {"preset": "production",
                                                "desc": "Multi-user server, <100K memories"},
                                 "research": {"preset": "research",
                                              "desc": "Maximum accuracy, unlimited memories"},
                                 "enterprise": {"preset": "enterprise",
                                                "desc": "Distributed system, 500K+ memories"},
                                 }},
                    {"step": 2,
                     "title": "Set Up Embedder",
                     "description": "Choose how to convert text to vectors",
                     "options": {"lm_studio": {"desc": "Local LM Studio (free, needs GPU)"},
                                 "openai": {"desc": "OpenAI API ($0.0001/request)"},
                                 "custom": {"desc": "Custom embedder function"},
                                 }},
                    {"step": 3,
                     "title": "Import Your Data",
                     "description": "Load existing knowledge into RTMDK",
                     "options": {"json": {"desc": "Import from JSON file"},
                                 "csv": {"desc": "Import from CSV file"},
                                 "manual": {"desc": "Start empty, add memories manually"},
                                 }},
                    {"step": 4,
                     "title": "Test & Verify",
                     "description": "Verify everything works",
                     "test_query": "What do you know?",
                     },
                    ]


class OnboardingWizard:
    """Guides new users through RTMDK setup.

    Usage:
        wizard = OnboardingWizard()
        steps = wizard.get_steps()
        # Guide user through each step
    """

    def __init__(self) -> None:
        self._completed_steps: Dict[int, Dict] = {}

    def get_steps(self) -> List[Dict]:
        """Get all onboarding steps."""
        return ONBOARDING_STEPS.copy()

    def complete_step(self, step: int, choices: Dict) -> None:
        """Mark a step as completed."""
        self._completed_steps[step] = choices

    def get_progress(self) -> Dict[str, Any]:
        """Get onboarding progress."""
        total = len(ONBOARDING_STEPS)
        completed = len(self._completed_steps)
        return {
            "total_steps": total,
            "completed_steps": completed,
            "progress_percent": round(completed / total * 100, 1),
            "next_step": completed + 1 if completed < total else None,
        }

    def get_recommended_config(self) -> Dict[str, Any]:
        """Get recommended config based on onboarding choices."""
        step1 = self._completed_steps.get(1, {})
        preset = step1.get("option", "local")

        return {
            "preset": preset,
            "next_steps": [
                "Install embedder (LM Studio or OpenAI)",
                "Create memory: create_rtmdk(preset, embedder)",
                "Import data: ImportPipeline(memory).import_json('data.json')",
                "Test: memory.load_memory_variables({'input': 'test'})",
            ]
        }
