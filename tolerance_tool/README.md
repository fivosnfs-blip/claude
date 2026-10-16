# Tolerance tool: lens-free SLM phase-retrieval head

A single script that computes alignment and sampling tolerances for a lens-free (Fresnel-zone) SLM phase-retrieval head, then plots how they change with pixel pitches and error budgets.

## 1. Install

```
pip install numpy matplotlib
```

## 2. Run

```
python tolerances.py          # prints the results table + warnings, runs the self-check, writes plots/*.png
python tolerances.py --check  # table + self-check only, no plots
```

Then open the PNGs in `plots/`.

The self-check tests the equations on a fixed copy of the default config with hand-computed results, so it keeps passing after you edit the config.

## 3. How to change things

Edit only the config block at the top of `tolerances.py` (SI units):

| Variable | Meaning |
|---|---|
| `wavelength` | laser wavelength (m) |
| `D` | beam 1/e² diameter on the SLM (m) |
| `p_slm` | SLM pixel pitch (m) |
| `cell_px` | mask cell size in SLM pixels; cell size `s = cell_px * p_slm` |
| `p_cam` | camera pixel pitch (m) |
| `sensor_px` | sensor pixel count (width, height); sensor size = count × `p_cam` |
| `z` | SLM-to-sensor distance (m) |
| `phi_tol` | phase error budget at the finest mask frequency (rad) |
| `reg_frac` | lateral registration budget as a fraction of one mask cell |
| `k_fit` | how finely the fit can locate the pattern, as a fraction of one camera pixel (typically 0.1–0.3) |
| `ne_ref` | mean electrons per pixel at the reference pitch `p_cam_ref` (signal level for the noise check) |
| `p_cam_ref` | pitch at which `ne_ref` applies (m) |
| `SHOW` | `True` also opens the plot windows |

Sweep ranges (for example `np.linspace(3.7e-6, 20e-6, 200)`) are on the first lines of each `plot_*` function.

## 4. How to read each plot

- **A, SLM pitch sweep:** z tolerance and registration budget grow with the SLM pitch (∝ s² and ∝ s), while the signal diameter shrinks. In the bottom-right panel, wherever the black pixel-order curve lies **below** the dashed D_sig/2 line, the pixel orders overlap the signal. Dotted verticals mark real SLMs.
- **B, camera pitch sweep (pixel count fixed):** larger pixels mean fewer ROI pixels but a larger sensor, so more margin. A margin below 0 means the signal does not fit. A Nyquist ratio above 1 means the pixel-order beat aliases.
- **C, Nyquist map:** red is aliased and blue is sampled correctly. The black line is `p_cam = p_slm/4`, and the star is your config.
- **D, phase budget:** dz_tol ∝ s²·phi_tol. Compare each line against what the mechanics (±500 µm) or a fitted z (±50 µm) can deliver.
- **E, registration budget:** lateral, rotation and scale tolerance vs reg_frac. Where a line goes flat, the camera floor `k_fit·p_cam` has taken over and loosening the budget no longer helps.
- **F, dz heatmap:** dz_tol over phi_tol × cell size. Configs to the right of the 50 µm or 500 µm contour meet that z accuracy.
- **G, registration floor:** each cell size is flat (budget-limited) until it meets the dashed camera floor `k_fit·p_cam`. Beyond that pitch, the camera limits registration and the line rises with the floor. Defaults: cell = 2 px meets the floor at about 8.3 µm, and cell = 1 px at about 4.2 µm.
- **H, box response:** the fraction of modulation a camera pixel keeps at that frequency. The mask structure stays near 1. The pixel-order beat (2/p_slm) is attenuated but **not** removed (0.57 at the defaults). If it is not small, the beat has to be in the forward model, and past the dotted `p_slm/4` line it also aliases.
- **I, noise check:** a rough shot-noise estimate of the phase noise per mask cell, compared with phi_tol. **This is a warning, not a tolerance.** If a line enters the shaded band, the budget is below the noise. The lines are flat in p_cam because, with the signal per pixel scaling with pixel area, the electrons per mask cell do not depend on pitch. Only cell size and `ne_ref` move them.

Camera pitch does **not** change dz_tol, which depends only on s and phi_tol. The camera enters through the registration floor (G), the sampling checks (B, C, H) and the noise check (I).

## 5. Derived vs chosen quantities

| Quantity | Type |
|---|---|
| `phi_tol`, `reg_frac` | **chosen** error budgets |
| `k_fit`, `ne_ref` | **assumptions** (fit precision, signal level) |
| s, D_sig, dz_tol, dx_budget, dx_floor, dx_eff, rotation/scale tol, tilt_max, x_pix, margin, W_min, ROI, Nyquist ratio, box response, sigma_phi_est | **derived** (physics/geometry from the inputs above) |

## 6. Adding your own sweep

`sweep(func, param_name, values, **fixed)` runs `func(config)` while one config entry varies. `summary(config)` returns every derived quantity by name.

```python
from tolerances import sweep, summary
import numpy as np, matplotlib.pyplot as plt
zs = np.linspace(40e-3, 120e-3, 100)
tilt = sweep(lambda c: np.degrees(summary(c)["tilt_max"]), "z", zs, cell_px=4)
plt.plot(zs * 1e3, tilt); plt.xlabel("z (mm)"); plt.ylabel("tilt_max (deg)"); plt.savefig("plots/my_sweep.png")
```
