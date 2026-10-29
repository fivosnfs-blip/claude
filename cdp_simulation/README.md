# CDP Beam Reconstruction: Simulation and Tolerance Study

`CDP_Simulation_v8_1_GPU_checked.ipynb` simulates **coded diffraction pattern (CDP) phase retrieval** of a laser beam. A spatial light modulator (SLM) puts `L` random phase masks on the beam one at a time. A camera at a short distance records the intensity pattern for each mask. The notebook then recovers the beam's complex field (amplitude and phase) from those images.

The notebook does two things:

1. **Single reconstruction**: it simulates one measurement and reconstructs the beam with spectral initialisation followed by accelerated Wirtinger Flow, then plots the true and recovered amplitude and phase.
2. **Tolerance sweeps**: it adds one error at a time to the system that generates the data (camera shift, distance, rotation, tilt, SLM zeroth order, noise, fill factor, beam aberrations, wavelength). It reconstructs with the nominal model and reports how large each error can be before the phase error passes **λ/20 rms**.

---

## Running it

The notebook is written for **Google Colab**:

1. Choose *Runtime → Change runtime type → T4 GPU* (optional, but much faster).
2. *Runtime → Run all*, or run cells 0–7, then the sweep settings cell, then any sweep cells.

**Dependencies:** `numpy`, `scipy`, `matplotlib`. On a GPU it also uses `cupy` (with `cupyx.scipy.sparse`), which Colab's GPU runtimes already include.

**GPU and CPU:** all heavy arrays live in `xp`. This is CuPy when a CUDA GPU is found and NumPy otherwise, so the same code runs on both. `to_cpu()` copies results back to NumPy for metrics and plots. Random numbers (masks, noise, start vectors) are always drawn on the CPU with NumPy and then moved to `xp`, so a given `seed` gives the same results on either device.

---

## Physical model

For mask `l`, the camera image is

```
y_l = S( P( conj(D_l) · X ) )        then camera noise
```

| Symbol | Meaning | Code |
|---|---|---|
| `X` | Complex beam on the SLM (the unknown), `n_slm × n_slm` | `define_sample` |
| `D_l` | Random phase mask with 256 levels, `exp(iφ)`. With a zeroth order: `√(1−η)·D + √η·e^{−iψ}` | `define_masks`, `effective_masks` |
| `P` | Band-limited angular-spectrum propagation over distance `z` | `Propagator` |
| `S` | Sampling onto camera pixels, then `|·|²` averaged over each pixel's area | `SamplingOperator` |

### `Propagator`: SLM to camera plane
- The SLM field is zero-padded to an `n_prop × n_prop` grid so light diffracted off the SLM does not wrap around (FFT periodicity) onto the camera.
- The transfer function includes the **Matsushima & Shimobaba (2009) band limit** and the **square-pixel sinc envelope** of the SLM, scaled by `fill_factor`.
- `upsample > 1` zero-pads the spectrum, so the field comes out on a finer grid (pitch `dx_slm / upsample`). This is exact band-limited interpolation between SLM-pitch samples.
- `forward` and `adjoint` are exact adjoints of each other, which the gradient relies on.

### `SamplingOperator`: propagation grid to camera
- Each camera pixel is sampled at `subsamples × subsamples` points over its area. The field at each point comes from **Keys cubic interpolation** (`cubic_kernel`) on the propagation grid. The pixel value is the mean intensity of its points.
- The camera can be **shifted**, **rotated** in-plane and **tilted** out of plane. Tilt is modelled as foreshortening plus an obliquity factor on the intensity. **The change in each pixel's distance to the SLM across a tilted camera is not modelled.**
- The whole mapping is stored as a single sparse matrix (cuSPARSE on the GPU), so `forward` and `adjoint` are each one sparse matrix product.
- At construction it prints a **warning if any camera pixel falls outside the propagation grid**. Those pixels read 0, which a real camera would not do. If you see the warning, increase `n_prop`.

### `choose_prop_grid`
This function picks the smallest power-of-two grid that:
- contains the SLM and the shifted, rotated camera;
- is wide enough that light diffracted at the SLM's largest angle cannot wrap onto the camera;
- samples the transfer function properly (`n ≥ λz / dx²`).

### Camera noise (`add_camera_noise`)
The exposure is set so the brightest pixel across all frames reaches `peak_fill × full_well` electrons. The function then applies Poisson shot noise and dark current, Gaussian read noise, full-well saturation, and ADC quantisation with a black-level offset. Finally it subtracts the offset and converts back to the units of `y`. Negative values are kept so the noise stays unbiased.

---

## Reconstruction algorithm

**Loss:** `f(X) = Σ_l ‖ |A_l X|² − y_l ‖²`, where `A_l = S ∘ P ∘ diag(conj(D_l))`.

1. **Spectral initialisation** (`spectral_init`): power iteration for the leading eigenvector of `Σ_l A_l* diag(y_l) A_l`. The result is scaled so its predicted images carry the same total intensity as the data. This is the counterpart, for this operator, of the λ² scaling in Candès et al.'s Wirtinger Flow. It stops when `1 − |⟨z_new, z⟩| < tol_SI`.
2. **Accelerated Wirtinger Flow** (`Wirtinger_iteration`):
   - Step size ramps up as `μ = min(1 − e^{−τ/τ₀}, mu_max)`.
   - With `normalize_step=True` the step is divided by a curvature scale computed from the data. This makes `mu_max` dimensionless, so the same value works across changes in brightness, optics and noise.
   - **Nesterov momentum** with an **adaptive restart** whenever the loss increases.
   - **Safeguard:** if a plain gradient step still increases the loss, the step is redone at half the length, so the iteration cannot diverge.
   - Stops when `‖∇f‖ / ‖X‖ < tol_WF` or after `iter_WF` iterations.

The beam can only be recovered **up to a global phase**. All comparisons first rotate the reconstruction onto the true beam.

---

## The scale model (`SCALE`)

A full-size simulation (2304-pixel camera, 6 mm beam, 15 cm distance) would be too large. The notebook therefore simulates a **scale model**:

- **Kept at real values:** SLM pitch (12.5 µm), camera pitch (6.5 µm), wavelength (631.8 nm), fill factor.
- **Divided by `SCALE`:** beam diameter, camera pixel count, SLM–camera distance.

This keeps the speckle size on the camera, the fraction of light that reaches the camera, and the ratio of measurements to unknowns the same as in the real setup. With the default `SCALE = 12` you get a 64 × 64 SLM region, a 192 × 192 camera, `z = 12.5 mm` and a 128 × 128 propagation grid (256 × 256 after upsampling). A smaller `SCALE` (8 or 6) is closer to the real setup and is affordable on a GPU, since the grid grows roughly as `1/SCALE`.

**Units in the sweeps are real-setup units.** Shift, distance, noise, zeroth order, fill factor and aberrations carry over unchanged. Rotation, tilt and wavelength errors are converted with:

| Function | Conversion | Why |
|---|---|---|
| `sim_rotation(deg)` | `deg × SCALE` | Moves the camera's edge pixels the same physical distance |
| `sim_tilt(deg)` | `arccos(1 − SCALE·(1 − cos θ))` | Gives the same edge foreshortening |
| `sim_wavelength_error(dλ)` | `dλ × SCALE` | Propagation depends on `λ·z`, and `z` is divided by `SCALE` |

According to the notebook, these conversions were checked against simulations at `SCALE` = 8, 12 and 16.

---

## Notebook layout

| Cell | Contents |
|---|---|
| 0 | Checks for a GPU (`nvidia-smi`) |
| 1 | **All model and solver code**: backend selection, `Propagator`, `SamplingOperator`, beam/mask/noise generation, `spectral_init`, `Wirtinger_iteration`, `reconstruct_x`, `plot2D` |
| 2 | **Parameters**: real setup, `SCALE`, solver settings, true vs. assumed system, zeroth order, detector model, precision, fill factor, camera noise. Builds the true system (`prop`, `sampler`) and the assumed system (`prop_ideal`, `sampler_ideal`). |
| 3 | Defines the true beam: Gaussian with defocus 5, astig 0° 2, coma x 1 (rad) |
| 4 | Generates the masks and camera images |
| 5 | Reconstructs and plots. Prints run times and the normalised overlap `|⟨x_r, x⟩| / (‖x_r‖‖x‖)` (labelled "Accuracy (%)", but printed as a fraction from 0 to 1) |
| 6–7 | `visualize_sampling_2d`: draws the camera pixels over the propagation grid and SLM aperture, to check the geometry |
| 8 | Sweep instructions (markdown) |
| 9 | **Sweep settings and helpers**, plus the perfect-system baseline run. Must be run before any sweep. |
| 10–18 | One sweep per cell (see below) |

### Two systems: "true" and "model"
Errors are modelled by building **two** systems. The *true* system (`*_true` parameters, `prop`, `sampler`) generates the data. The *model* system (nominal parameters, `prop_ideal`, `sampler_ideal`) is what the reconstruction assumes. Making them differ, for example `shift_x_true = 5e-6`, simulates a misaligned or miscalibrated setup.

---

## Tolerance sweeps

Every point in a sweep is a full, independent simulation: generate data with one error, reconstruct with the nominal model, compare with the true beam. `n_seeds` repeats each point with different masks, noise and start vector. Plots show the mean with min–max error bars, the λ/20 limit, and the perfect-system level.

| Cell | Sweep | Range (real units) |
|---|---|---|
| 10 | Lateral camera shift, x and y | 1–6 µm |
| 11 | SLM–camera distance, camera farther or closer | 0.05–0.6 mm |
| 12 | In-plane camera rotation | 0.025–0.3° |
| 13 | Camera tilt about x, about y, and about both (also reports the induced astigmatism bias) | 1–6° |
| 14 | SLM zeroth-order leak: ignored by the model / known / known power with phase off by 90° | 1–10 % power |
| 15 | Noise and exposure: electrons in the brightest pixel, full well 2 000 and 20 000 e⁻ | 20 – 18 000 e⁻ |
| 16 | Fill-factor mismatch (model assumes 0.968) | true value 0.80–1.00 |
| 17 | Beam aberrations, one term varied at a time (defocus, astig 0°, coma x). Flags points where the phase changes by more than π/2 between neighbouring SLM pixels, since results there are not meaningful. | 0–30 rad |
| 18 | Wavelength error. The top axis shows the equivalent distance error `z·Δλ/λ`. | ±0.2, ±0.5 nm |

### Metrics (`_metrics`)
- **`phase_rms`**: amplitude-weighted (`|x_true|²`) rms phase error in rad, after removing the global phase. This is the pass/fail metric. The limit is `phase_rms_limit = 2π/20 ≈ 0.314 rad`, which is λ/20, about 32 nm of optical path difference.
- **`rel_err`**: `‖x_r − x_true‖ / ‖x_true‖` after aligning the global phase.
- **`bias_<term>`**: weighted least-squares fit of the phase error onto the low-order aberration terms. It shows *which* aberration an error introduces, for example tilt → astigmatism.
- **`iters`**: Wirtinger Flow iterations used.

`tolerance()` linearly interpolates where the mean `phase_rms` curve first crosses the limit. If it never does within the range, it reports "> max (never fails)". `convergence_check()` warns about NaN (failed) runs, and about **passing** runs that stopped at the iteration limit. In that case, increase `SWEEP["iter_WF"]` to confirm the tolerance.

### Sweep settings (`SWEEP` in cell 9)
| Setting | Default | Note |
|---|---|---|
| `QUICK` | `False` | `True`: 1 seed and `tol_WF = 1e-3`, about 3× faster, for a first look |
| `n_seeds` | 2 | Independent repeats per point |
| `upsample` | 2 | Keep at 2. The Fourier upsampling is what makes the detector model accurate. |
| `pixel_subsamples` | 1 | About 2.5× faster than the 2 used in the main run. Skips averaging over each pixel's area. |
| `tol_WF` | 3e-4 | Looser than the main run's 1e-4. About 25 % fewer iterations with the same tolerances. |
| `precision` | `"single"` | Matches double precision at the plotted accuracy |
| `n_workers` | 1 on GPU, all cores on CPU | CPU runs use a `fork` process pool. The GPU runs in one process because CUDA does not survive forking. |

---

## Key parameters (cell 2)

| Parameter | Default | Meaning |
|---|---|---|
| `SCALE` | 12 | Scale-model factor (see above) |
| `z_real`, `beam_diameter_real`, `n_cam_real` | 0.15 m, 6 mm (1/e²), 2304 | Real setup |
| `dx_slm`, `dx_cam` | 12.5 µm, 6.5 µm | SLM and camera pixel pitch |
| `L` | 16 | Number of masks |
| `wavelength` | 631.8 nm | |
| `iter_WF`, `tol_WF`, `tau0`, `mu_max` | 1000, 1e-4, 20, 0.2 | Wirtinger Flow |
| `iter_SI`, `tol_SI` | 500, 1e-3 | Spectral initialisation |
| `*_true` | equal to nominal | Errors in the true system (wavelength, z, shift, rotation, tilt) |
| `zero_order_power[_true]`, `zero_order_phase[_true]` | 0 | Unmodulated SLM light (assumed by the model / present in the data) |
| `fill_factor[_true]` | 0.968 | SLM fill factor (assumed by the model / present in the data) |
| `upsample`, `pixel_subsamples` | 2, 2 | Detector-model accuracy |
| `precision` | `"single"` | Keep single on the GPU. A T4 runs double precision about 30× slower. |
| `use_noise`, `camera_noise` | `True`; 20 000 e⁻ full well, 80 % peak fill, 3 e⁻ read noise, 12-bit ADC, 100 ADU offset | Camera model from the datasheet |
| `seed` | 0 | Makes masks, noise and start vectors reproducible |

---

## Things to know before changing the code

- **Re-run order:** after editing cell 2, re-run cells 3–5 and **cell 9** before any sweep. Cell 9 captures `NOMINAL` and the baseline from the current parameters.
- **Keep `base_aberrations` (cell 9) equal to the beam in cell 3**, or the sweeps and the main run will describe different beams.
- `reconstruct_x(..., plot_x=True)` plots against the **global** variable `x`, not an argument. Its `sigma` argument is unused and is kept only for compatibility.
- `generate_measurement` returns the **programmed** masks `D`. The reconstruction should be given `effective_masks(D, zero_order_power, zero_order_phase)`, i.e. the masks the model *believes* the SLM produced, as cell 5 does.
- The SLM region is 1.6× the beam diameter (`n_slm` in cell 2). Strong aberrations can make the phase change faster than the SLM grid can represent. Cell 17's `max_phase_step` checks for this.
- On the CPU, sweeps use `multiprocessing` with the `fork` start method. If that is not available, they fall back to sequential runs and print a message.

---

## References

- E. J. Candès, X. Li, M. Soltanolkotabi, "Phase retrieval from coded diffraction patterns," *Appl. Comput. Harmon. Anal.* 39 (2015).
- E. J. Candès, X. Li, M. Soltanolkotabi, "Phase retrieval via Wirtinger Flow: theory and algorithms," *IEEE Trans. Inf. Theory* 61 (2015).
- K. Matsushima, T. Shimobaba, "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields," *Opt. Express* 17 (2009).
- R. Keys, "Cubic convolution interpolation for digital image processing," *IEEE Trans. Acoust., Speech, Signal Process.* 29 (1981).
- B. O'Donoghue, E. Candès, "Adaptive restart for accelerated gradient schemes," *Found. Comput. Math.* 15 (2015).
