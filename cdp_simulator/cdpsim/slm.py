"""SLM masks: what is commanded (saved in phi) and what the LC layer actually applies."""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import SimConfig

TWO_PI = 2.0 * math.pi


def generate_masks(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Random commanded masks, uniform over the 2^bits phase levels, constant over
    mask_block x mask_block pixel blocks. Returns (K, My, Mx) float32 radians in [0, 2pi)."""
    levels = 2 ** cfg.slm_bits
    b = int(cfg.mask_block)
    My, Mx = cfg.slm_shape
    g = rng.integers(0, levels, size=(cfg.n_masks, -(-My // b), -(-Mx // b)))
    g = np.repeat(np.repeat(g, b, axis=1), b, axis=2)[:, :My, :Mx]
    return (g * (TWO_PI / levels)).astype(np.float32)


def quantize_phase(phi: np.ndarray, bits: int = 8) -> np.ndarray:
    """Round a phase map to the nearest SLM level, result in [0, 2pi)."""
    levels = 2 ** bits
    g = np.mod(np.round(np.asarray(phi) / TWO_PI * levels), levels)
    return g * (TWO_PI / levels)


def walkoff_margin(cfg: SimConfig) -> int:
    """SLM pixels around a window that the phase blurs need to see."""
    m = 0
    if cfg.crosstalk:
        m += int(math.ceil(4 * cfg.crosstalk_sigma_px)) + 1
    if cfg.walkoff:
        m += 1
    return m


def applied_phase(phi_cmd: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Phase the LC layer applies, on the sub-pixel grid (oversample x oversample per pixel).

    Order: wrap to the displayable range -> phase response (nonlinearity / short range)
    -> fringing-field crosstalk -> LC walk-off. With every switch off this returns the
    commanded phase (wrapped to [0, 2pi)) repeated onto the sub-grid.
    """
    s = int(cfg.oversample)
    phi = np.mod(np.asarray(phi_cmd, dtype=np.float64), TWO_PI)
    if cfg.phase_nonlinearity:
        phi = cfg.phase_range * (phi / TWO_PI) ** cfg.phase_gamma
    if s > 1:
        phi = np.repeat(np.repeat(phi, s, axis=0), s, axis=1)
    if cfg.crosstalk:
        # The LC director relaxes smoothly between pixels driven to different levels.
        # The blur acts on the physical retardance, so a 2pi -> 0 step is a real large step.
        phi = gaussian_filter(phi, sigma=cfg.crosstalk_sigma_px * s, mode="nearest")
    if cfg.walkoff:
        # Light crosses the LC layer obliquely, so it averages the retardance over a box of
        # width dx along x. For dx < q (sub-sample pitch) the cell average of the boxcar-
        # blurred piecewise-constant profile is a 3-tap filter [c, 1-2c, c], c = dx / (8 q).
        q = cfg.slm_pitch / s
        if cfg.walkoff_dx >= q:
            raise ValueError("walk-off model needs walkoff_dx < slm_pitch / oversample")
        c = cfg.walkoff_dx / (8.0 * q)
        padded = np.pad(phi, ((0, 0), (1, 1)), mode="edge")
        phi = phi + c * (padded[:, 2:] + padded[:, :-2] - 2.0 * phi)
    return phi


def pixel_mean_field(phi_sub: np.ndarray, s: int) -> np.ndarray:
    """Average of exp(i phi) over each SLM pixel (s x s sub-samples)."""
    e = np.exp(1j * phi_sub)
    if s == 1:
        return e
    My, Mx = phi_sub.shape[0] // s, phi_sub.shape[1] // s
    return e.reshape(My, s, Mx, s).mean(axis=(1, 3))


def applied_phase_native(phi_cmd: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Applied phase reduced to one value per SLM pixel (for truth.npz / mismatch studies)."""
    s = int(cfg.oversample)
    return np.mod(np.angle(pixel_mean_field(applied_phase(phi_cmd, cfg), s)), TWO_PI)
