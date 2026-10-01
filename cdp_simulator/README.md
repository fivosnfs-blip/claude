# CDP simulator — Part 1 (forward model only)

Simulates the off-axis coded-diffraction set-up (HeNe → Hamamatsu X15213-01 SLM at
θ = 5° → f = 100 mm Fourier lens → 3.45 µm, 2448 × 2048, 12-bit camera) and writes the
camera images for a set of random SLM masks. It does **no phase retrieval**.

```bash
pip install -r requirements.txt
python -m cdpsim.simulate --out data --masks 8          # ideal set-up
python -m cdpsim.report data                            # data/report.png + data/report.txt
python -m pytest -q                                     # physics + data-contract tests (~40 s)

# model mismatch: any combination of switches (all off by default)
python -m cdpsim.simulate --out data_ni --oversample 4 --pixel-gaps --walkoff --tilted-plane \
    --phase-nonlinearity --crosstalk --lens-clipping --camera-defocus --focal-length-error
```

On 4 CPU cores the optics for one full-sensor image take ~0.3 s. Including noise and the
compressed save, a run takes ~4 s per mask. With `--oversample 4` and lens clipping it takes ~15 s per mask.

## Data contract

`measurements.npz` (what Part 2 reads)

| key | shape | content |
|---|---|---|
| `I` | (K, Ny, Nx) float32 | camera images in ADU after shot noise, read noise, offset, full well, 12-bit quantisation; Ny × Nx is the ROI (full sensor by default) |
| `phi` | (K, 1024, 1272) float32 | **commanded** SLM phases in radians, multiples of 2π/256 in [0, 2π) |
| `meta` | 0-d str | JSON, read with `json.loads(str(d["meta"]))` |

`meta` holds only what an experimenter would know: `wavelength, f, slm_pitch, slm_shape,
slm_bits, slm_axis_pixel, cam_pitch, cam_shape, cam_axis_pixel, roi [y0, x0, ny, nx],
mask_block, n_masks, bits_cam, units, gain (ADU/e⁻), offset (ADU), read_noise (e⁻), qe,
full_well, beam_center_on_slm, photons_per_unit (per image), freq_step, sigma_oversampling,
spectrum_width, pixel_convention, forward_model`. Hidden imperfections are **not** in `meta`.

`truth.npz` (scoring only): `u_true` (1024, 1272) complex field on the SLM pixel grid,
`phi_applied` (the phases the LC layer actually applied, one per pixel), `sim_config` and
`beam_config` (JSON, including the switches), `diagnostics` (JSON), and `I_clean`
(noise-free images) if `--save-clean` was given.

Conventions: SLM pixel `(i, j)` is centred at `((i, j) − slm_axis_pixel) · slm_pitch`; camera
pixel `(k, l)` (full-sensor index) sees spatial frequency
`((k, l) − cam_axis_pixel) · cam_pitch / (λ f)`. `u_true` is normalised so
`Σ|u|² p² = 1`, so the noise-free image is the fraction of beam power per camera pixel and

```
E(f)  = p² sinc(p fx) sinc(p fy) · Σ_n u_n e^{iφ_n} e^{−2πi f·x_n}
I_ADU ≈ offset + gain · qe · photons_per_unit · |E(f)|² · (cam_pitch / λf)²
```

## How the optics are modelled

* **SLM field.** Each pixel is a square cell with constant field (sampled at the cell centre,
  or on s × s sub-cells with `--oversample s`). The exact Fourier transform of that field is a
  DTFT times the pixel-aperture envelope `S(f) = p² sinc(p fx) sinc(p fy)`. The DTFT is periodic
  in 1/p, so the higher diffraction orders appear on the sensor (λf/p = 5.06 mm < 8.45 mm), weighted
  by the envelope.
* **Camera sampling.** The frequency step per camera pixel, `cam_pitch / (λ f)` = 54.52 m⁻¹, is
  computed from the parameters. No integer FFT length gives that step (it would need N = 1467.4),
  so the DTFT is evaluated directly at the camera frequencies with a separable matrix DFT
  (`Ey @ t @ Ex`). This is exact resampling, with no interpolation, and costs about as much as an FFT of the same size.
* **SLM → lens distance d** only matters when lens clipping is on. In the back focal plane it
  only adds a phase that the camera cannot see.

### Switches (all off by default)

| switch | model |
|---|---|
| `pixel_gaps` (`fill_factor` 0.968) | the modulated area is a square of side p√FF; the gaps reflect the beam unmodulated, which adds a zero-order term `(S_p − S_a)·DTFT{u}` |
| `walkoff` (0.55 µm) | the LC retardance is averaged over a box of width Δx along x. For a piecewise-constant profile this is the 3-tap filter `[c, 1−2c, c]`, c = Δx/(8q) |
| `tilted_plane` (θ = 5°) | sample x sits at x cos θ transversally and x sin θ deeper along the axis. Each column gets its own Fresnel propagation `exp(−iπλ x sinθ (fx²+fy²))`, which keeps the transform separable |
| `phase_nonlinearity` | applied phase = range · (φ/2π)^γ, with range = 0.92·2π and γ = 1.15 |
| `crosstalk` (σ = 0.42 px, FWHM ≈ 1 px) | Gaussian blur of the retardance (fringing fields), so a 2π→0 flyback is a real step. Use with `--oversample 4` |
| `lens_clipping` (R = 11 mm) | spectrum on a 2× finer frequency grid → exact angular-spectrum propagation over d → circular pupil → FFT back to the camera |
| `camera_defocus` (Δz = 0.2 mm) | ABCD matrix of d → lens → f+Δz: quadratic phase `exp(−iπ Δz r²/(λ f B))` on the SLM field, scale B = f + Δz(1 − d/f) |
| `focal_length_error` (+1 %) | the true f is used for the simulation; `meta` keeps the nominal f |
| `pixel_integration` | the camera pixel averages intensity over its 3.45 µm area (sinc MTF, about 17 % contrast loss at the finest speckle). Not in the original list, so off as well |

Commanded masks (`phi`) and applied masks (`truth.phi_applied`) are kept separate. The
order is: wrap to [0, 2π) → phase response → crosstalk → walk-off.

### Camera

The exposure is set so the brightest expected pixel reaches `peak_fraction` (0.8) of full scale. By default one exposure
is shared by all K images (`--exposure per_image` scales each image on its own). The model then applies
Poisson shot noise on `qe · photons`, a 10 500 e⁻ full well, 2.5 e⁻ Gaussian read noise,
a 32 ADU offset, gain = 4095 / full well, rounding, and clipping to 12 bits.

## Tests (`tests/test_physics.py`)

* **Flat mask gives the Fourier transform of the beam only.** The image matches πw² e^{−π²w²f²} (× pixel sinc²)
  to 3·10⁻⁵ of the peak, which is the residual from the SLM edge truncating the Gaussian. There is no light beyond 6 spot radii, and
  the power on the sensor is 1 − p²/(6w²), the predicted share that the pixel envelope leaves in the higher orders.
* **A quadratic phase on the beam spreads the spot** as Gaussian optics predicts (10⁻⁴), and on an off-axis beam
  it shifts the centroid by −f·x₀/R (2·10⁻⁴). A linear phase shifts the spot by λ f a.
* **Doubling the field multiplies the intensity by 4**, both noise-free (10⁻⁵) and through the
  camera at fixed exposure.
* **Parseval.** The on-sensor power for a random mask equals the sinc²-envelope prediction (0.5 %).
* The matrix DFT matches a brute-force DTFT. Camera defocus matches Gaussian-beam propagation.
  A focal-length error rescales the pattern by 1+ε, and the tilted plane widens it by 1/cos θ.
  The clipping path is neutral for an unclipped beam. Pixel gaps create a zero order.
  Sub-pixel sampling converges. The masks are quantised and blocked, and the data contract is met.
