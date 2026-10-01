"""Incoming beam: Gaussian amplitude, smooth Zernike phase, optional vortex."""
from __future__ import annotations

import numpy as np

from .config import BeamConfig, SimConfig

# Orthonormal (RMS = 1 over the unit disc) Zernike terms, rho = r / norm_radius.
ZERNIKE = {
    "tilt_x": lambda r, t: 2 * r * np.cos(t),
    "tilt_y": lambda r, t: 2 * r * np.sin(t),
    "defocus": lambda r, t: np.sqrt(3) * (2 * r**2 - 1),
    "astig_0": lambda r, t: np.sqrt(6) * r**2 * np.cos(2 * t),
    "astig_45": lambda r, t: np.sqrt(6) * r**2 * np.sin(2 * t),
    "coma_x": lambda r, t: np.sqrt(8) * (3 * r**3 - 2 * r) * np.cos(t),
    "coma_y": lambda r, t: np.sqrt(8) * (3 * r**3 - 2 * r) * np.sin(t),
    "spherical": lambda r, t: np.sqrt(5) * (6 * r**4 - 6 * r**2 + 1),
}


class Beam:
    """Callable field u(Y, X) on the SLM surface, coordinates in metres from the optical axis.

    The field is normalised so that sum |u|^2 * pitch^2 over the SLM pixel grid equals
    `power`, which makes the camera images directly comparable to the beam power.
    """

    def __init__(self, beam: BeamConfig, cfg: SimConfig):
        for name in beam.zernike:
            if name not in ZERNIKE:
                raise ValueError(f"unknown Zernike term {name!r}; choose from {sorted(ZERNIKE)}")
        self.beam = beam
        p = cfg.slm_pitch
        ay, ax = cfg.slm_axis()
        cy, cx = beam.center if beam.center is not None else (ay, ax)
        self.center_px = (float(cy), float(cx))
        self.y0 = (cy - ay) * p
        self.x0 = (cx - ax) * p
        self.scale = 1.0
        My, Mx = cfg.slm_shape
        yv = (np.arange(My) - ay) * p
        xv = (np.arange(Mx) - ax) * p
        u = self(yv[:, None], xv[None, :])
        self.scale = np.sqrt(beam.power / (np.sum(np.abs(u) ** 2) * p * p))

    def __call__(self, Y, X):
        b = self.beam
        y = Y - self.y0
        x = X - self.x0
        r2 = x * x + y * y
        w = b.diameter / 2.0
        amp = np.exp(-r2 / w**2)
        phase = np.zeros(np.broadcast(x, y).shape)
        rho = np.sqrt(r2) / b.norm_radius
        th = np.arctan2(y, x)
        for name, c in b.zernike.items():
            if c:
                phase = phase + c * ZERNIKE[name](rho, th)
        if b.vortex_charge:
            amp = amp * (np.sqrt(r2) / w) ** abs(b.vortex_charge)
            phase = phase + b.vortex_charge * th
        return self.scale * amp * np.exp(1j * phase)
