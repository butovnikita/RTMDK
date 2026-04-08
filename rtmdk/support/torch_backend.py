"""Torch backend for RTMDK."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    pass


class TorchBackend:
    def __init__(self):
        self.torch = None
        self.device = None
        try:
            import torch
            self.torch = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self.torch is not None

    def batch_resonance(self, ql, qp, np_, nph, na, ns, bw, pc):
        if not self.available:
            return self._numpy(ql, qp, np_, nph, na, ns, bw, pc)
        tq = self.torch.from_numpy(ql).to(self.device)
        dists = self.torch.cdist(tq, self.torch.from_numpy(np_).to(self.device))
        spatial = self.torch.exp(-dists / bw)
        pd = qp.unsqueeze(1) - self.torch.from_numpy(nph).to(self.device).unsqueeze(0)
        pa = 0.5 + 0.5 * self.torch.cos(pd)
        r = spatial * ((1 - pc) + pc * pa)
        return (r * self.torch.from_numpy(na).to(self.device).unsqueeze(0) * self.torch.from_numpy(ns).to(self.device).unsqueeze(0)).cpu().numpy()

    @staticmethod
    def _numpy(ql, qp, np_, nph, na, ns, bw, pc):
        dists = cdist(ql, np_)
        spatial = np.exp(-dists / bw)
        pd = qp[:, np.newaxis] - nph[np.newaxis, :]
        pa = 0.5 + 0.5 * np.cos(pd)
        return spatial * ((1 - pc) + pc * pa) * na[np.newaxis, :] * ns[np.newaxis, :]
