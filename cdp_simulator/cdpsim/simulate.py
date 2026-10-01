"""Part 1 driver: simulate the coded-diffraction measurements and write the data files.

    python -m cdpsim.simulate --out data --masks 8
    python -m cdpsim.simulate --out data_nonideal --tilted-plane --crosstalk --oversample 4

measurements.npz : I (K, Ny, Nx) ADU, phi (K, My, Mx) commanded radians, meta (JSON string)
truth.npz        : u_true (My, Mx) complex, phi_applied, sim_config, beam_config, diagnostics
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import camera as cam_mod
from .beam import Beam
from .config import BeamConfig, SimConfig, sampling_budget
from .optics import ForwardModel
from .slm import applied_phase_native, generate_masks


def build_meta(cfg: SimConfig, beam: Beam, model: ForwardModel, scales: np.ndarray) -> dict:
    """What Part 2 is allowed to know: nominal set-up values, no hidden imperfections."""
    cam = cfg.camera
    budget = sampling_budget(cfg, beam.beam)
    return {
        "wavelength": cfg.wavelength,
        "f": cfg.f,
        "slm_pitch": cfg.slm_pitch,
        "slm_shape": list(cfg.slm_shape),
        "slm_bits": cfg.slm_bits,
        "slm_axis_pixel": list(cfg.slm_axis()),
        "cam_pitch": cam.pitch,
        "cam_shape": list(cam.shape),
        "cam_axis_pixel": list(cam.axis()),
        "roi": list(cfg.roi_box()),
        "mask_block": cfg.mask_block,
        "n_masks": cfg.n_masks,
        "bits_cam": cam.bits,
        "units": "ADU",
        "gain": cam.adu_per_e,
        "offset": cam.offset,
        "read_noise": cam.read_noise,
        "qe": cam.qe,
        "full_well": cam.full_well,
        "beam_center_on_slm": list(beam.center_px),
        "photons_per_unit": [float(v) for v in scales],
        "freq_step": model.freq_step_nominal,
        "sigma_oversampling": budget["sigma_oversampling"],
        "spectrum_width": budget["spectrum_width_m"],
        "pixel_convention": "pixel index i has its centre at coordinate (i - axis_pixel) * pitch",
        "forward_model": (
            "E(f) = p^2 sinc(p fx) sinc(p fy) * sum_n u_n exp(i phi_n) exp(-2 pi i f.x_n), "
            "x_n = (n - slm_axis_pixel) * slm_pitch, f = (k - cam_axis_pixel) * cam_pitch / "
            "(wavelength f); I_ADU ~ offset + gain * qe * photons_per_unit * |E|^2 * freq_step^2"
        ),
    }


def simulate(cfg: SimConfig, beam_cfg: BeamConfig, out_dir: str | Path | None = None,
             save_clean: bool = False, verbose: bool = True) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    beam = Beam(beam_cfg, cfg)
    model = ForwardModel(cfg, beam)
    phi = generate_masks(cfg, rng)
    K = cfg.n_masks
    ny, nx = cfg.roi_box()[2:]
    if verbose:
        y0, y1, x0, x1 = model.crop
        print(f"SLM crop {y1 - y0} x {x1 - x0} px (oversample {cfg.oversample}), "
              f"camera ROI {ny} x {nx} px, freq step {model.freq_step:.3f} /m")

    I_clean = np.empty((K, ny, nx), dtype=np.float32)
    for k in range(K):
        I_clean[k] = model.intensity(phi[k])
        if verbose:
            print(f"  mask {k + 1}/{K}: on-sensor power {I_clean[k].sum(dtype=np.float64):.4f}")

    scales = cam_mod.exposure_scales(I_clean, cfg.camera)
    I = np.empty_like(I_clean)
    stats = []
    for k in range(K):
        I[k], st = cam_mod.expose(I_clean[k].astype(np.float64), scales[k], cfg.camera, rng)
        st["power_on_sensor"] = float(I_clean[k].sum(dtype=np.float64))
        stats.append(st)

    meta = build_meta(cfg, beam, model, scales)
    diagnostics = {
        "slm_power": model.slm_power,
        "per_image": stats,
        "sampling": sampling_budget(cfg, beam_cfg),
        "freq_step_true": model.freq_step,
        "max_beam_phase_step_rad": model.max_phase_step,
        "slm_crop": list(model.crop),
        "runtime_s": None,
    }
    result = {"I": I, "phi": phi, "meta": meta, "u_true": model.u_true,
              "I_clean": I_clean, "diagnostics": diagnostics, "model": model}

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        phi_applied = np.stack([applied_phase_native(phi[k], cfg) for k in range(K)])
        diagnostics["runtime_s"] = time.time() - t0
        np.savez_compressed(out / "measurements.npz", I=I, phi=phi, meta=np.array(json.dumps(meta)))
        extra = {"I_clean": I_clean} if save_clean else {}
        np.savez_compressed(
            out / "truth.npz", u_true=model.u_true, phi_applied=phi_applied.astype(np.float32),
            sim_config=np.array(json.dumps(cfg.to_dict())),
            beam_config=np.array(json.dumps(asdict(beam_cfg))),
            diagnostics=np.array(json.dumps(diagnostics)), **extra)
        if verbose:
            print(f"wrote {out / 'measurements.npz'} and {out / 'truth.npz'} "
                  f"in {time.time() - t0:.1f} s")
    return result


def load_measurements(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    d = np.load(path)
    return d["I"], d["phi"], json.loads(str(d["meta"]))


def _parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--masks", type=int, default=8)
    ap.add_argument("--mask-block", type=int, default=1, choices=(1, 2))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oversample", type=int, default=1)
    ap.add_argument("--beam-diameter", type=float, default=4e-3)
    ap.add_argument("--vortex", type=int, default=0)
    ap.add_argument("--flat-phase", action="store_true", help="no Zernike aberrations")
    ap.add_argument("--peak-fraction", type=float, default=0.8)
    ap.add_argument("--exposure", default="global", choices=("global", "per_image"))
    ap.add_argument("--roi", type=int, nargs=4, metavar=("Y0", "X0", "NY", "NX"))
    ap.add_argument("--save-clean", action="store_true", help="store noise-free images in truth.npz")
    for name in ("pixel_gaps", "walkoff", "tilted_plane", "phase_nonlinearity", "crosstalk",
                 "lens_clipping", "camera_defocus", "focal_length_error", "pixel_integration"):
        ap.add_argument("--" + name.replace("_", "-"), action="store_true")
    ap.add_argument("--defocus-dz", type=float, default=0.2e-3)
    ap.add_argument("--focal-error", type=float, default=0.01)
    return ap.parse_args(argv)


def main(argv=None):
    a = _parse(argv)
    cfg = SimConfig(n_masks=a.masks, mask_block=a.mask_block, seed=a.seed, oversample=a.oversample,
                    roi=tuple(a.roi) if a.roi else None,
                    defocus_dz=a.defocus_dz, focal_error=a.focal_error)
    for name in ("pixel_gaps", "walkoff", "tilted_plane", "phase_nonlinearity", "crosstalk",
                 "lens_clipping", "camera_defocus", "focal_length_error", "pixel_integration"):
        setattr(cfg, name, getattr(a, name))
    cfg.camera.peak_fraction = a.peak_fraction
    cfg.camera.exposure = a.exposure
    beam = BeamConfig(diameter=a.beam_diameter, vortex_charge=a.vortex)
    if a.flat_phase:
        beam.zernike = {}
    budget = sampling_budget(cfg, beam)
    if budget["sigma_oversampling"] < 2:
        print(f"WARNING: camera oversampling sigma = {budget['sigma_oversampling']:.2f} < 2")
    if beam.diameter * 1.5 > cfg.slm_shape[0] * cfg.slm_pitch:
        print("WARNING: 1.5 x beam diameter exceeds the SLM short side")
    simulate(cfg, beam, a.out, save_clean=a.save_clean)


if __name__ == "__main__":
    main()
