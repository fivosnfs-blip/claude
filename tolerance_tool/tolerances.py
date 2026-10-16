"""Tolerance explorer for a lens-free SLM phase-retrieval head (Fresnel zone, no lens).

Run:  python tolerances.py          -> prints the current config, runs the self-check, saves plots/*.png
      python tolerances.py --check  -> prints the current config and runs the self-check only
"""
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm
from matplotlib.ticker import FuncFormatter

# ============================ CONFIG (edit here) ============================
wavelength = 632.8e-9      # m, laser wavelength
D = 6e-3                   # m, beam 1/e^2 diameter on the SLM
p_slm = 12.5e-6            # m, SLM pixel pitch
cell_px = 2                # mask cell size in SLM pixels (s = cell_px * p_slm)
p_cam = 3.45e-6            # m, camera pixel pitch
sensor_px = (4096, 3000)   # camera pixels (width, height); sensor size = px * p_cam
z = 69.03e-3               # m, SLM to sensor distance
phi_tol = 0.04             # rad, phase error budget at the finest mask frequency
reg_frac = 0.1             # registration budget as a fraction of one mask cell
k_fit = 0.3                # fraction of a camera pixel to which the fit locates the pattern (0.1..0.3)
ne_ref = 1500              # mean electrons per pixel at the reference pitch
p_cam_ref = 3.45e-6        # m, pitch at which ne_ref applies
SHOW = False               # True -> also open plot windows
# ===========================================================================

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
UM, MM = 1e6, 1e3          # SI -> micrometres / millimetres (for printing and plotting only)
REAL_SLMS = {"Holoeye LETO": 6.4e-6, "Holoeye PLUTO": 8.0e-6, "Meadowlark": 9.2e-6, "Hamamatsu": 12.5e-6}
# One colour per cell size, used in every plot; SLM pitches get their own colours (dashed lines)
CELL_COLORS = {1: "#008300", 2: "#2a78d6", 3: "#eb6834", 4: "#1baf7a", 6: "#e87ba4"}
SLM_COLORS = {6.4e-6: "#6250d6", 9.2e-6: "#e34948", 12.5e-6: "#eda100"}
if not SHOW:
    matplotlib.use("Agg")


def config():
    """Current config block as a dict (so studies can override single values)."""
    return dict(wavelength=wavelength, D=D, p_slm=p_slm, cell_px=cell_px, p_cam=p_cam,
                sensor_px=sensor_px, z=z, phi_tol=phi_tol, reg_frac=reg_frac, k_fit=k_fit,
                ne_ref=ne_ref, p_cam_ref=p_cam_ref)


# ============================ EQUATIONS ============================
def cell_size(cell_px, p_slm):             # s = cell_px * p_slm
    return cell_px * p_slm
def f_max(s):                              # f_max = 1 / (2 s)
    return 1 / (2 * s)
def signal_diameter(D, z, wavelength, s):  # D_sig = D + 2 z wavelength / (2 s)
    return D + 2 * z * wavelength / (2 * s)
def dz_tolerance(s, phi_tol, wavelength):  # dz_tol = 4 s^2 phi_tol / (pi wavelength)
    return 4 * s**2 * phi_tol / (np.pi * wavelength)
def dx_budget(reg_frac, s):                # dx_budget = reg_frac * s
    return reg_frac * s
def tilt_max(dz_tol, D_sig):               # tilt_max = arctan(dz_tol / (D_sig / 2))
    return np.arctan(dz_tol / (D_sig / 2))
def pixel_order_pos(z, wavelength, p_slm): # x_pix = z tan(wavelength / p_slm)
    return z * np.tan(wavelength / p_slm)
def sensor_margin(sensor_h, D_sig):        # margin per side = (sensor_height - D_sig) / 2
    return (sensor_h - D_sig) / 2
def window_min(D_sig, x_pix):              # W_min = D_sig + 2 x_pix
    return D_sig + 2 * x_pix
def roi_pixels(D_sig, p_cam):              # ROI = D_sig / p_cam
    return D_sig / p_cam
def nyquist_ratio(p_cam, p_slm):           # p_cam / (p_slm / 4) > 1 -> beat 2/p_slm aliases
    return p_cam / (p_slm / 4)
def dx_floor(k_fit, p_cam):                # dx_floor = k_fit * p_cam
    return k_fit * p_cam
def rot_scale_tol(dx_eff, D):              # rot_tol = scale_tol = dx_eff / (D / 2)
    return dx_eff / (D / 2)
def box_response(f, p_cam):                # attenuation = sinc(f p_cam), numpy sinc includes pi
    return np.sinc(f * p_cam)

def sigma_phi_est(s, p_cam, ne_ref, p_cam_ref):
    # sigma_phi ~ 1/sqrt(ne * (s/p_cam)^2), ne = ne_ref (p_cam/p_cam_ref)^2.
    # ROUGH order-of-magnitude shot-noise estimate only; a warning, not a derived tolerance.
    ne = ne_ref * (p_cam / p_cam_ref)**2
    return 1 / np.sqrt(ne * (s / p_cam)**2)

# Note: camera pitch does NOT change dz_tol (it depends only on s and phi_tol).
# The camera enters through the registration floor, the sampling checks and the noise estimate.


def summary(c):
    """All derived quantities for one config dict c."""
    r = {}
    r["s"] = cell_size(c["cell_px"], c["p_slm"])
    r["f_max"] = f_max(r["s"])
    r["D_sig"] = signal_diameter(c["D"], c["z"], c["wavelength"], r["s"])
    r["dz_tol"] = dz_tolerance(r["s"], c["phi_tol"], c["wavelength"])
    r["dx_budget"] = dx_budget(c["reg_frac"], r["s"])
    r["dx_floor"] = dx_floor(c["k_fit"], c["p_cam"])
    r["dx_eff"] = max(r["dx_budget"], r["dx_floor"])
    r["camera_limited"] = r["dx_budget"] < r["dx_floor"]
    r["rot_tol"] = r["scale_tol"] = rot_scale_tol(r["dx_eff"], c["D"])
    r["tilt_max"] = tilt_max(r["dz_tol"], r["D_sig"])
    r["x_pix"] = pixel_order_pos(c["z"], c["wavelength"], c["p_slm"])
    r["sensor_h"] = c["sensor_px"][1] * c["p_cam"]
    r["margin"] = sensor_margin(r["sensor_h"], r["D_sig"])
    r["W_min"] = window_min(r["D_sig"], r["x_pix"])
    r["roi_px"] = roi_pixels(r["D_sig"], c["p_cam"])
    r["pitch_ratio"] = c["p_slm"] / c["p_cam"]
    r["nyquist_ratio"] = nyquist_ratio(c["p_cam"], c["p_slm"])
    r["f_nyq"] = 1 / (2 * c["p_cam"])
    r["box_mask"] = box_response(1 / r["s"], c["p_cam"])
    r["box_beat"] = box_response(2 / c["p_slm"], c["p_cam"])
    r["sigma_phi"] = sigma_phi_est(r["s"], c["p_cam"], c["ne_ref"], c["p_cam_ref"])
    return r


def sweep(func, param_name, values, **fixed):
    """Evaluate func(config) while one config entry runs over values. Returns a numpy array."""
    out = []
    for v in values:
        c = config()
        c.update(fixed)
        c[param_name] = v
        out.append(func(c))
    return np.array(out)


def print_summary(c):
    r = summary(c)
    rows = [("cell size s", r["s"] * UM, "um"), ("signal diameter D_sig", r["D_sig"] * MM, "mm"),
            ("z tolerance dz_tol", r["dz_tol"] * UM, "um"), ("registration budget dx_budget", r["dx_budget"] * UM, "um"),
            ("camera floor dx_floor", r["dx_floor"] * UM, "um"), ("effective lateral dx_eff", r["dx_eff"] * UM, "um"),
            ("rotation tolerance", np.degrees(r["rot_tol"]), "deg"), ("scale tolerance", r["scale_tol"] * 100, "%"),
            ("sensor tilt max", np.degrees(r["tilt_max"]), "deg"), ("pixel-order position x_pix", r["x_pix"] * MM, "mm"),
            ("window W_min", r["W_min"] * MM, "mm"), ("sensor margin per side", r["margin"] * MM, "mm"),
            ("ROI across signal", r["roi_px"], "px"), ("pitch ratio p_slm/p_cam", r["pitch_ratio"], ""),
            ("Nyquist ratio p_cam/(p_slm/4)", r["nyquist_ratio"], ""),
            ("box response @ mask f=1/s", r["box_mask"], ""), ("box response @ beat f=2/p_slm", r["box_beat"], ""),
            ("sigma_phi_est (rough)", r["sigma_phi"], "rad"), ("phi_tol (budget)", c["phi_tol"], "rad")]
    print(f"\nConfig: p_slm={c['p_slm']*UM:g} um, cell_px={c['cell_px']}, p_cam={c['p_cam']*UM:g} um, z={c['z']*MM:g} mm")
    print("-" * 52)
    for name, val, unit in rows:
        print(f"{name:32s} {val:12.4g} {unit}")
    print("-" * 52)
    if r["margin"] < 0:
        print("WARNING: signal does not fit on the sensor short side")
    if r["x_pix"] < r["D_sig"] / 2:
        print("WARNING: SLM pixel orders overlap the signal")
    if r["nyquist_ratio"] > 1:
        print(f"WARNING: Nyquist ratio {r['nyquist_ratio']:.2f} > 1, pixel-order beat aliases "
              f"(2/p_slm = {2/c['p_slm']/1e3:.0f}/mm > Nyquist {r['f_nyq']/1e3:.0f}/mm)")
    if r["camera_limited"]:
        print("WARNING: registration is camera-limited (dx_budget < k_fit * p_cam)")
    if c["phi_tol"] < r["sigma_phi"]:
        print("WARNING: phi_tol is below the rough noise estimate sigma_phi_est")
    return r


def self_check():
    """Check the equations against hand-computed numbers for a fixed reference config (independent of edits above)."""
    ref = dict(wavelength=632.8e-9, D=6e-3, p_slm=12.5e-6, cell_px=2, p_cam=3.45e-6, sensor_px=(4096, 3000),
               z=69.03e-3, phi_tol=0.04, reg_frac=0.1, k_fit=0.3, ne_ref=1500, p_cam_ref=3.45e-6)
    r, r4 = summary(ref), summary(dict(ref, cell_px=4))
    expected = [("s [um]", r["s"] * UM, 25, 0.02), ("D_sig [mm]", r["D_sig"] * MM, 7.75, 0.005),
                ("dz_tol [um]", r["dz_tol"] * UM, 50, 0.02), ("dx_budget [um]", r["dx_budget"] * UM, 2.5, 0.01),
                ("dx_floor [um]", r["dx_floor"] * UM, 1.04, 0.01), ("dx_eff [um]", r["dx_eff"] * UM, 2.5, 0.01),
                ("rot_tol [mrad]", r["rot_tol"] * 1e3, 0.83, 0.01), ("rot_tol [deg]", np.degrees(r["rot_tol"]), 0.048, 0.01),
                ("scale_tol [%]", r["scale_tol"] * 100, 0.083, 0.01), ("tilt_max [deg]", np.degrees(r["tilt_max"]), 0.74, 0.01),
                ("x_pix [mm]", r["x_pix"] * MM, 3.50, 0.005), ("margin [mm]", r["margin"] * MM, 1.30, 0.01),
                ("Nyquist ratio", r["nyquist_ratio"], 1.10, 0.01), ("box @ mask", r["box_mask"], 0.97, 0.01),
                ("box @ beat", r["box_beat"], 0.57, 0.01),
                ("D_sig cell_px=4 [mm]", r4["D_sig"] * MM, 6.87, 0.005), ("dz_tol cell_px=4 [um]", r4["dz_tol"] * UM, 200, 0.02)]
    for name, got, want, rtol in expected:
        assert abs(got - want) <= rtol * abs(want), f"{name}: got {got:.4g}, want {want:.4g}"
    assert not r["camera_limited"], "reference config should not be camera-limited"
    print("Self-check passed (all asserts OK).")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, name), dpi=150)
    print("saved plots/" + name)
    if not SHOW:
        plt.close(fig)


def style(ax, title, xlabel, ylabel, legend_loc="best"):
    """Title, axis labels, grid and legend for one subplot."""
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc=legend_loc)


def plain_log_ticks(axis, ticks):
    """Label a log axis with plain numbers at the given ticks (instead of 2x10^-2)."""
    axis.set_ticks(ticks)
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def mark_current(ax, x, label="current"):
    ax.axvline(x, color="k", lw=1, ls="-.", alpha=0.6, label=label)


# ============================ STUDY 1: pixel pitches ============================
def plot_A_slm_pitch():
    pitches = np.linspace(3.7e-6, 20e-6, 200)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    sensor_h = sensor_px[1] * p_cam
    for n in (2, 4):
        col = CELL_COLORS[n]
        dz = sweep(lambda c: summary(c)["dz_tol"], "p_slm", pitches, cell_px=n)
        dx = sweep(lambda c: summary(c)["dx_budget"], "p_slm", pitches, cell_px=n)
        dsig = sweep(lambda c: summary(c)["D_sig"], "p_slm", pitches, cell_px=n)
        axs[0, 0].plot(pitches * UM, dz * UM, color=col, lw=2, label=f"cell = {n} px")
        axs[0, 1].plot(pitches * UM, dx * UM, color=col, lw=2, label=f"cell = {n} px")
        axs[1, 0].plot(pitches * UM, dsig * MM, color=col, lw=2, label=f"cell = {n} px")
        axs[1, 1].plot(pitches * UM, dsig / 2 * MM, color=col, lw=2, ls="--", label=f"D_sig/2, cell = {n} px")
    xpix = pixel_order_pos(z, wavelength, pitches)
    axs[1, 1].plot(pitches * UM, xpix * MM, color="k", lw=2, label="pixel order x_pix")
    axs[1, 0].axhline(sensor_h * MM, color="0.3", ls="--", label=f"sensor height {sensor_h*MM:.2f} mm")
    titles = ["z tolerance", "Lateral registration budget", "Signal diameter on sensor", "SLM pixel-order position"]
    ylabels = ["dz_tol (µm)", "dx_budget (µm)", "D_sig (mm)", "distance from axis (mm)"]
    for ax, t, yl in zip(axs.flat, titles, ylabels):
        for name, p in REAL_SLMS.items():
            ax.axvline(p * UM, color="0.5", ls=":", lw=1)
            ax.text(p * UM, 0.98, f" {name} {p*UM:.1f}", rotation=90, va="top", ha="right",
                    fontsize=7, color="0.35", transform=ax.get_xaxis_transform())
        style(ax, t, "SLM pixel pitch (µm)", yl, "center right")
    fig.suptitle(f"SLM pitch sweep (z = {z*MM:.2f} mm, phi_tol = {phi_tol} rad, reg_frac = {reg_frac})")
    save(fig, "A_slm_pitch.png")


def plot_B_cam_pitch():
    cams = np.linspace(2e-6, 8e-6, 200)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ps, col in SLM_COLORS.items():
        lab = f"p_slm = {ps*UM:.1f} µm"
        roi = sweep(lambda c: summary(c)["roi_px"], "p_cam", cams, p_slm=ps)
        marg = sweep(lambda c: summary(c)["margin"], "p_cam", cams, p_slm=ps)
        axs[0].plot(cams * UM, roi, color=col, lw=2, label=lab)
        axs[1].plot(cams * UM, marg * MM, color=col, lw=2, label=lab)
        axs[2].plot(cams * UM, nyquist_ratio(cams, ps), color=col, lw=2, label=lab)
    axs[1].axhline(0, color="k", lw=1, ls="--", label="signal just fits")
    axs[2].axhline(1, color="k", lw=1, ls="--", label="aliasing limit")
    titles = ["ROI across the signal", "Sensor margin per side (short side)", "Nyquist ratio of the pixel-order beat"]
    ylabels = ["ROI (camera pixels)", "margin (mm)", "p_cam / (p_slm/4)"]
    for ax, t, yl in zip(axs, titles, ylabels):
        mark_current(ax, p_cam * UM)
        style(ax, t, "camera pixel pitch (µm)", yl)
    fig.suptitle(f"Camera pitch sweep, sensor fixed at {sensor_px[0]} x {sensor_px[1]} px, cell = {cell_px} px")
    save(fig, "B_cam_pitch.png")


def plot_C_nyquist_map():
    ps = np.linspace(3.7e-6, 20e-6, 200)
    pc = np.linspace(1e-6, 10e-6, 200)
    PS, PC = np.meshgrid(ps, pc)
    ratio = nyquist_ratio(PC, PS)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.pcolormesh(PS * UM, PC * UM, ratio, cmap="coolwarm", shading="auto",
                       norm=TwoSlopeNorm(vmin=ratio.min(), vcenter=1, vmax=ratio.max()))
    cs = ax.contour(PS * UM, PC * UM, ratio, levels=[1], colors="k", linewidths=2.5)
    ax.clabel(cs, fmt="ratio = 1 (aliasing limit)", fontsize=8)
    ax.plot(p_slm * UM, p_cam * UM, "k*", ms=14, mfc="w", label="current config")
    cbar = fig.colorbar(im, ax=ax, format="%g", label="Nyquist ratio p_cam / (p_slm/4)   (>1 = beat aliases)")
    cbar.set_ticks([0.25, 0.5, 0.75, 1, 2, 4, 6, 8, 10])
    ax.set(title="Pixel-order beat sampling: red = aliased", xlabel="SLM pixel pitch (µm)",
           ylabel="camera pixel pitch (µm)")
    ax.legend(loc="upper right")
    save(fig, "C_nyquist_map.png")


def plot_G_registration_floor():
    cams = np.linspace(1e-6, 10e-6, 300)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for n in (1, 2, 3, 4):
        dx = sweep(lambda c: summary(c)["dx_eff"], "p_cam", cams, cell_px=n)
        ax.plot(cams * UM, dx * UM, color=CELL_COLORS[n], lw=2, label=f"cell = {n} px")
    ax.plot(cams * UM, dx_floor(k_fit, cams) * UM, "k--", lw=1.5, label=f"camera floor {k_fit} x p_cam")
    mark_current(ax, p_cam * UM, f"current p_cam = {p_cam*UM:.2f} µm")
    ax.set_yscale("log")
    plain_log_ticks(ax.yaxis, [0.3, 0.5, 1, 2, 3, 5])
    style(ax, f"Effective lateral registration tolerance (p_slm = {p_slm*UM:.1f} µm, reg_frac = {reg_frac})",
          "camera pixel pitch (µm)", "lateral tolerance dx_eff (µm)", "lower right")
    save(fig, "G_registration_floor.png")


def plot_H_box_response():
    cams = np.linspace(1e-6, 10e-6, 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in (2, 4):
        s = cell_size(n, p_slm)
        ax.plot(cams * UM, np.abs(box_response(1 / s, cams)), color=CELL_COLORS[n], lw=2,
                label=f"mask f = 1/s, cell = {n} px (s = {s*UM:.0f} µm)")
    for ps, col in SLM_COLORS.items():
        ax.plot(cams * UM, np.abs(box_response(2 / ps, cams)), color=col, lw=2, ls="--",
                label=f"beat f = 2/p_slm, p_slm = {ps*UM:.1f} µm")
        ax.axvline(ps / 4 * UM, color=col, lw=1, ls=":")
        ax.text(ps / 4 * UM, 0.03, f" alias {ps*UM:.1f}", rotation=90, fontsize=7, color=col, va="bottom")
    mark_current(ax, p_cam * UM, f"current p_cam = {p_cam*UM:.2f} µm")
    ax.set_ylim(0, 1.02)
    style(ax, "Camera pixel box response |sinc(f p_cam)|  (dotted: p_cam = p_slm/4)",
          "camera pixel pitch (µm)", "attenuation |sinc| (1 = no loss)", "upper right")
    save(fig, "H_box_response.png")


def plot_I_noise_check():
    cams = np.linspace(1e-6, 10e-6, 300)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    top = 0
    for n in (2, 4):
        sig = sigma_phi_est(cell_size(n, p_slm), cams, ne_ref, p_cam_ref)
        top = max(top, sig.max())
        ax.plot(cams * UM, sig, color=CELL_COLORS[n], lw=2, label=f"sigma_phi_est, cell = {n} px")
    ax.axhline(phi_tol, color="k", lw=1.5, label=f"phi_tol = {phi_tol} rad")
    ax.axhspan(phi_tol, max(top, phi_tol) * 3, color="#e34948", alpha=0.12, label="budget below noise")
    mark_current(ax, p_cam * UM, f"current p_cam = {p_cam*UM:.2f} µm")
    ax.set_yscale("log")
    style(ax, "Noise sanity check (ROUGH order-of-magnitude estimate, not a tolerance)",
          "camera pixel pitch (µm)", "phase noise per mask cell (rad)", "upper right")
    ax.text(0.02, 0.5, f"ne = {ne_ref} e-/px at {p_cam_ref*UM:.2f} µm, scaled with pixel area.\n"
            "Flat in p_cam: electrons collected per mask cell do not depend on pitch.",
            transform=ax.transAxes, fontsize=8, color="0.3")
    save(fig, "I_noise_check.png")


# ============================ STUDY 2: error budgets ============================
def plot_D_phase_budget():
    phis = np.logspace(np.log10(0.01), np.log10(0.5), 200)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for n in (2, 3, 4, 6):
        dz = sweep(lambda c: summary(c)["dz_tol"], "phi_tol", phis, cell_px=n)
        ax.plot(phis, dz * UM, color=CELL_COLORS[n], lw=2, label=f"cell = {n} px")
    ax.axhline(500, color="0.3", ls="--", lw=1.2)
    ax.axhline(50, color="0.3", ls=":", lw=1.2)
    ax.text(0.011, 560, "mechanics sets z to ±500 µm", fontsize=8, color="0.3")
    ax.text(0.011, 56, "fitted z ±50 µm", fontsize=8, color="0.3")
    ax.plot(phi_tol, dz_tolerance(cell_size(cell_px, p_slm), phi_tol, wavelength) * UM, "k*", ms=13,
            mfc="w", label=f"default (phi_tol {phi_tol}, cell {cell_px} px)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_ticks(ax.xaxis, [0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    style(ax, f"z tolerance vs phase budget (p_slm = {p_slm*UM:.1f} µm)", "phase budget phi_tol (rad)", "dz_tol (µm)")
    save(fig, "D_phase_budget.png")


def plot_E_registration_budget():
    fracs = np.logspace(np.log10(0.02), np.log10(0.3), 200)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for n in (2, 3, 4, 6):
        dx = sweep(lambda c: summary(c)["dx_eff"], "reg_frac", fracs, cell_px=n)
        rot = rot_scale_tol(dx, D)
        lab = f"cell = {n} px"
        axs[0].plot(fracs, dx * UM, color=CELL_COLORS[n], lw=2, label=lab)
        axs[1].plot(fracs, np.degrees(rot), color=CELL_COLORS[n], lw=2, label=lab)
        axs[2].plot(fracs, rot * 100, color=CELL_COLORS[n], lw=2, label=lab)
    titles = ["Lateral tolerance", "Rotation tolerance", "Scale tolerance"]
    ylabels = ["dx_eff (µm)", "rotation (degrees)", "scale (%)"]
    for ax, t, yl in zip(axs, titles, ylabels):
        mark_current(ax, reg_frac, f"default reg_frac = {reg_frac}")
        ax.set_xscale("log")
        plain_log_ticks(ax.xaxis, [0.02, 0.05, 0.1, 0.2, 0.3])
        style(ax, t, "registration budget reg_frac (fraction of a cell)", yl)
    fig.suptitle(f"Registration budget sweep (flat parts = camera floor {k_fit} x p_cam = "
                 f"{dx_floor(k_fit, p_cam)*UM:.2f} µm)")
    save(fig, "E_registration_budget.png")


def plot_F_dz_heatmap():
    phis = np.logspace(np.log10(0.01), np.log10(0.5), 200)
    cells = np.arange(1, 9)
    PHI, CELL = np.meshgrid(phis, cells)
    dz = dz_tolerance(cell_size(CELL, p_slm), PHI, wavelength) * UM
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.pcolormesh(PHI, CELL, dz, cmap="Blues", shading="nearest",
                       norm=LogNorm(vmin=dz.min(), vmax=dz.max()))
    cs = ax.contour(PHI, CELL, dz, levels=[50, 500], colors="k", linestyles=[":", "--"], linewidths=2)
    ax.clabel(cs, fmt={50: "50 µm", 500: "500 µm"}, fontsize=9)
    ax.plot(phi_tol, cell_px, "k*", ms=13, mfc="w", label="default")
    fig.colorbar(im, ax=ax, label="dz_tol (µm)")
    ax.set_xscale("log")
    plain_log_ticks(ax.xaxis, [0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    ax.set(title=f"z tolerance map (p_slm = {p_slm*UM:.1f} µm)", xlabel="phase budget phi_tol (rad)",
           ylabel="mask cell size (SLM pixels)", yticks=cells)
    ax.legend(loc="upper left")
    save(fig, "F_dz_heatmap.png")


if __name__ == "__main__":
    print_summary(config())
    self_check()
    if "--check" not in sys.argv:
        os.makedirs(PLOT_DIR, exist_ok=True)
        for plot in (plot_A_slm_pitch, plot_B_cam_pitch, plot_C_nyquist_map, plot_G_registration_floor,
                     plot_H_box_response, plot_I_noise_check, plot_D_phase_budget,
                     plot_E_registration_budget, plot_F_dz_heatmap):
            plot()
        if SHOW:
            plt.show()
