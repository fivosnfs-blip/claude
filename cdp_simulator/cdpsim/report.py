"""Report on a simulated data set.

    python -m cdpsim.report data            -> data/report.png and data/report.txt

Shows the true amplitude and phase, one measured pattern on a log scale with the
mask spectrum width lambda f / p drawn against the sensor, photon count and SNR per
image, and power conservation (Parseval) between the SLM and camera planes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .beam import Beam  # noqa: E402
from .config import SimConfig, beam_from_dict, sampling_budget  # noqa: E402
from .optics import ForwardModel  # noqa: E402


def parseval(model: ForwardModel, cfg: SimConfig, phi0: np.ndarray) -> dict:
    """Power on the SLM vs power on the sensor, flat mask and first random mask."""
    p_slm = model.slm_power
    flat = float(model.intensity(None).sum())
    I0 = model.intensity(phi0)
    rand = float(I0.sum())
    # For independent uniform phases on b x b blocks, E|E(f)|^2 = |S_bp(f)|^2 sum|u|^2 / (b p)^2 * ...
    # i.e. the on-sensor fraction is the integral of the block sinc^2 envelope over the sensor.
    yc, xc = model.camera_coords()
    lam_f = cfg.wavelength * cfg.f
    bp = cfg.mask_block * cfg.slm_pitch
    pred = ((np.sinc(bp * yc / lam_f) ** 2).sum() * (np.sinc(bp * xc / lam_f) ** 2).sum()
            * (bp * model.freq_step) ** 2)
    return {"slm_power": p_slm, "flat_on_sensor": flat, "flat_fraction": flat / p_slm,
            "random_on_sensor": rand, "random_fraction": rand / p_slm,
            "random_fraction_predicted": float(pred), "I_clean0": I0}


def make_report(data_dir: str | Path, mask_index: int = 0) -> str:
    data_dir = Path(data_dir)
    m = np.load(data_dir / "measurements.npz")
    t = np.load(data_dir / "truth.npz")
    meta = json.loads(str(m["meta"]))
    cfg = SimConfig.from_dict(json.loads(str(t["sim_config"])))
    beam_cfg = beam_from_dict(json.loads(str(t["beam_config"])))
    diag = json.loads(str(t["diagnostics"]))
    I, phi, u = m["I"], m["phi"], t["u_true"]
    k = mask_index

    beam = Beam(beam_cfg, cfg)
    model = ForwardModel(cfg, beam)
    pv = parseval(model, cfg, phi[k])
    budget = sampling_budget(cfg, beam_cfg)
    cam = cfg.camera
    ppu = meta["photons_per_unit"][k]

    # ------------------------------------------------------------------ text
    lines = [
        f"Data set: {data_dir}",
        f"K = {I.shape[0]} masks, mask block {cfg.mask_block}x{cfg.mask_block}, "
        f"camera ROI {meta['roi']} -> images {I.shape[1]} x {I.shape[2]}",
        "",
        "Sampling",
        f"  frequency step per camera pixel  cam_pitch/(lambda f) = {budget['freq_step_per_m']:.3f} cycles/m",
        f"  oversampling sigma = lambda f/(cam_pitch D_ext) = {budget['sigma_oversampling']:.2f} "
        f"(D_ext = {budget['D_ext_m'] * 1e3:.1f} mm, need >= 2)",
        f"  spectrum width lambda f/(b p) = {budget['spectrum_width_m'] * 1e3:.2f} mm "
        f"({budget['spectrum_width_px']:.0f} px) vs sensor "
        f"{budget['sensor_width_m'] * 1e3:.2f} x {budget['sensor_height_m'] * 1e3:.2f} mm "
        f"-> {'fits' if budget['spectrum_fits_sensor'] else 'DOES NOT FIT'}",
        f"  speckle size lambda f/D_ext = {budget['speckle_size_m'] * 1e6:.1f} um",
        f"  largest beam phase step between SLM pixels = {diag['max_beam_phase_step_rad']:.3f} rad",
        "",
        "Photons and SNR per image (energy SNR = ||signal|| / ||noise||)",
        "   k   photons on sensor   electrons   peak e-   SNR pred   SNR meas   saturated px",
    ]
    for i, st in enumerate(diag["per_image"]):
        lines.append(f"  {i:2d}   {st['photons_on_sensor']:16.4e}   {st['electrons_on_sensor']:9.3e}"
                     f"   {st['peak_electrons']:7.0f}   {st['snr_predicted']:8.1f}   "
                     f"{st['snr_measured']:8.1f}   {st['saturated_pixels']:6d}")
    lines += [
        "",
        "Power conservation (Parseval), noise-free model",
        f"  SLM plane:  sum |u|^2 p^2                  = {pv['slm_power']:.6f}",
        f"  flat mask:  power on sensor                = {pv['flat_on_sensor']:.6f}"
        f"  ({100 * pv['flat_fraction']:.4f} %)",
        f"  mask {k}:     power on sensor                = {pv['random_on_sensor']:.6f}"
        f"  ({100 * pv['random_fraction']:.2f} %)",
        f"  ideal-SLM prediction (sinc^2 envelope)   = {pv['random_fraction_predicted']:.6f}"
        f"  ({100 * pv['random_fraction_predicted']:.2f} %)",
        "  The rest of the random-mask light is in the higher diffraction orders outside the sensor.",
    ]
    text = "\n".join(lines)

    # ------------------------------------------------------------------ figure
    p = cfg.slm_pitch
    ay, ax = cfg.slm_axis()
    by, bx = beam.center_px
    half = int(round(1.5 * beam_cfg.diameter / p))
    ys = slice(max(int(by) - half, 0), min(int(by) + half + 2, u.shape[0]))
    xs = slice(max(int(bx) - half, 0), min(int(bx) + half + 2, u.shape[1]))
    ext_slm = [(xs.start - ax) * p * 1e3, (xs.stop - ax) * p * 1e3,
               (ys.stop - ay) * p * 1e3, (ys.start - ay) * p * 1e3]
    uc = u[ys, xs]
    amp = np.abs(uc)
    ph = np.where(amp > 0.05 * amp.max(), np.angle(uc), np.nan)

    yc, xc = model.camera_coords()
    ext_cam = [xc[0] * 1e3, xc[-1] * 1e3, yc[-1] * 1e3, yc[0] * 1e3]
    img = I[k] - meta["offset"]

    fig, axs = plt.subplots(2, 3, figsize=(17, 10.5))
    a = axs[0, 0]
    im = a.imshow(amp, extent=ext_slm, cmap="magma")
    a.set_title("True amplitude |u| on the SLM")
    a.set_xlabel("x (mm)"); a.set_ylabel("y (mm)")
    fig.colorbar(im, ax=a, shrink=0.8)
    a = axs[0, 1]
    im = a.imshow(ph, extent=ext_slm, cmap="twilight", vmin=-math.pi, vmax=math.pi)
    a.set_title("True phase arg(u) (rad), where |u| > 5% of max")
    a.set_xlabel("x (mm)")
    fig.colorbar(im, ax=a, shrink=0.8)

    a = axs[0, 2]
    im = a.imshow(np.log10(np.maximum(img, 1.0)), extent=ext_cam, cmap="inferno")
    a.set_title(f"Measured pattern, mask {k}: log10(ADU - offset)")
    a.set_xlabel("x camera (mm)"); a.set_ylabel("y camera (mm)")
    w = budget["spectrum_width_m"] * 1e3
    a.add_patch(Rectangle((-w / 2, -w / 2), w, w, fill=False, ec="cyan", ls="--", lw=1.2,
                          label=f"lambda f / p = {w:.2f} mm"))
    sx, sy = cam.shape[1] * cam.pitch * 1e3, cam.shape[0] * cam.pitch * 1e3
    cay, cax = cam.axis()
    a.add_patch(Rectangle((-cax * cam.pitch * 1e3, -cay * cam.pitch * 1e3), sx, sy, fill=False,
                          ec="white", lw=1.5, label=f"sensor {sx:.2f} x {sy:.2f} mm"))
    a.legend(loc="lower right", fontsize=8)
    fig.colorbar(im, ax=a, shrink=0.8)

    a = axs[1, 0]
    row = int(round(cay - meta["roi"][0]))
    expected = cam.adu_per_e * cam.qe * ppu * pv["I_clean0"][row]
    a.semilogy(xc * 1e3, np.maximum(expected, 0.5), lw=0.6, color="C3", label="expected (noise-free)")
    a.semilogy(xc * 1e3, np.maximum(img[row], 0.5), ".", ms=1.5, color="k", alpha=0.6,
               label="measured")
    a.axvline(-w / 2, color="C0", ls="--", lw=0.8)
    a.axvline(w / 2, color="C0", ls="--", lw=0.8, label="+- lambda f / 2p")
    a.set_title(f"Row through the optical axis, mask {k}")
    a.set_xlabel("x camera (mm)"); a.set_ylabel("ADU - offset")
    a.legend(fontsize=8)

    a = axs[1, 1]
    st = diag["per_image"]
    kk = np.arange(len(st))
    a.bar(kk, [s["photons_on_sensor"] for s in st], 0.5, color="C1", label="photons on sensor")
    a.set_ylabel("photons per image"); a.set_xlabel("mask k"); a.set_xticks(kk)
    a2 = a.twinx()
    a2.plot(kk, [s["snr_measured"] for s in st], "o-", color="C0", label="SNR measured")
    a2.plot(kk, [s["snr_predicted"] for s in st], "x--", color="C2", label="SNR predicted")
    a2.set_ylabel("energy SNR")
    a2.set_ylim(0, 1.2 * max(s["snr_predicted"] for s in st))
    a.set_title("Photon count and SNR per image")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")

    a = axs[1, 2]
    a.axis("off")
    a.text(0, 1, "\n".join(lines[3:9] + [""] + lines[-7:-1]), va="top", family="monospace",
           fontsize=7.2, transform=a.transAxes)
    fig.suptitle("CDP simulator report", fontsize=14)
    fig.tight_layout()
    fig.savefig(data_dir / "report.png", dpi=110)
    plt.close(fig)
    (data_dir / "report.txt").write_text(text + "\n")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("data_dir", nargs="?", default="data")
    ap.add_argument("--mask", type=int, default=0)
    a = ap.parse_args(argv)
    print(make_report(a.data_dir, a.mask))
    print(f"\nwrote {Path(a.data_dir) / 'report.png'}")


if __name__ == "__main__":
    main()
