"""Camera model: exposure, shot noise, read noise, dark offset, full well, 12-bit ADC."""
from __future__ import annotations

import numpy as np

from .config import CameraConfig


def photons_for_peak(peak_power: float, cam: CameraConfig) -> float:
    """Photons per unit power such that the brightest expected pixel sits at
    peak_fraction of full scale (offset included)."""
    target_e = (cam.peak_fraction * cam.full_scale - cam.offset) / cam.adu_per_e
    return target_e / (cam.qe * peak_power)


def exposure_scales(I_clean: np.ndarray, cam: CameraConfig) -> np.ndarray:
    """Photons per unit power for each of the K images."""
    K = I_clean.shape[0]
    if cam.exposure == "fixed":
        if cam.photons_per_unit is None:
            raise ValueError("exposure='fixed' needs photons_per_unit")
        return np.full(K, float(cam.photons_per_unit))
    if cam.exposure == "global":
        return np.full(K, photons_for_peak(float(I_clean.max()), cam))
    if cam.exposure == "per_image":
        return np.array([photons_for_peak(float(I_clean[k].max()), cam) for k in range(K)])
    raise ValueError(f"unknown exposure mode {cam.exposure!r}")


def expose(I: np.ndarray, photons_per_unit: float, cam: CameraConfig,
           rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Turn a noise-free power image into a recorded image in ADU (float32)."""
    g = cam.adu_per_e
    mean_e = cam.qe * photons_per_unit * I
    e = rng.poisson(mean_e).astype(np.float64) if cam.shot_noise else mean_e.copy()
    e = np.minimum(e, cam.full_well)                    # pixel saturation
    if cam.read_noise:
        e += rng.normal(0.0, cam.read_noise, size=e.shape)
    adu = g * e + cam.offset
    if cam.quantize:
        adu = np.round(adu)
    adu = np.clip(adu, 0, cam.full_scale)               # ADC range

    expected = g * np.minimum(mean_e, cam.full_well) + cam.offset
    var_e = mean_e * cam.shot_noise + cam.read_noise**2 + (1.0 / 12.0) / g**2 * cam.quantize
    signal = float(np.sqrt(np.sum(mean_e**2)))
    stats = {
        "photons_on_sensor": float(photons_per_unit * I.sum()),
        "electrons_on_sensor": float(mean_e.sum()),
        "peak_electrons": float(mean_e.max()),
        "peak_adu_expected": float(g * mean_e.max() + cam.offset),
        # Energy SNR of the whole image: ||signal|| / ||noise|| (predicted and measured)
        "snr_predicted": signal / float(np.sqrt(np.sum(var_e))),
        "snr_measured": float(np.linalg.norm(expected - cam.offset)
                              / max(np.linalg.norm(adu - expected), 1e-30)),
        "peak_pixel_snr": float(mean_e.max() / np.sqrt(mean_e.max() + cam.read_noise**2)),
        "saturated_pixels": int(np.count_nonzero(adu >= cam.full_scale)),
    }
    return adu.astype(np.float32), stats
