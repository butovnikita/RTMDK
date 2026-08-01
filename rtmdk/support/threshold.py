"""Adaptive threshold for RTMDK."""

from __future__ import annotations

from collections import deque

import numpy as np


class AdaptiveThreshold:
    def __init__(self, window_size: int = 30, base_threshold: float = 0.25, sensitivity: float = 0.5):
        self.window: deque = deque(maxlen=window_size)
        self.base_threshold = base_threshold
        self.sensitivity = sensitivity
        self.current_threshold = base_threshold

    def record_tension(self, tension: float):
        self.window.append(tension)
        if len(self.window) >= 5:
            self.current_threshold = float(
                max(0.01, np.mean(self.window) + self.sensitivity * np.std(self.window))
            )  # type: ignore[arg-type]

    def get_threshold(self) -> float:
        return self.current_threshold
