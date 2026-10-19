# CDP_Simulation v7.1 → v7.2: what changed, where and why

Cells are numbered 1–7 from the top of the notebook. Every code change in
cell 1 is marked with a `# [FIX n]` comment so you can search for it.

| Cell | Contents | Changed? |
|---|---|---|
| 1 | classes and functions | yes (fixes 1, 2, 3, 5) |
| 2 | parameters | yes (fixes 1, 4, 5) |
| 3 | `x = define_sample(...)` | no |
| 4 | `generate_measurement(...)` | yes (fix 5) |
| 5 | reconstruction loop | yes (fixes 4, 5) |
| 6 | `visualize_sampling_2d` definition | yes (fixes 1, 3) |
| 7 | `visualize_sampling_2d(...)` call | yes (fixes 1, 3, 4) |

---

## Fix 1: zero-padded, band-limited propagation grid

**Problem.** The angular-spectrum propagation ran on the 45 × 45 SLM grid
(1.62 mm). The FFT treats that window as periodic, so light diffracted out of
it wrapped round and came back in on the opposite side. At z = 0.2 m about
78 % of the light leaves the window. The camera (1.92 mm) was also larger than
the window, so 2044 of 6400 pixels always read 0. On top of that, the transfer
function H was undersampled: the grid needs N ≥ λz/dx² ≈ 98, and it had 45.

**Changes.**

1. **Cell 1, `class Propagator`:**
   - The constructor is now `Propagator(n_in, n, dx, wavelength, z)`. It was
     `Propagator(n, dx, wavelength, z)`.
     - `n_in` is the SLM grid size.
     - `n` is the larger propagation grid the SLM is zero-padded into.
   - New attributes:
     - `offset`, where the SLM sits inside the grid.
     - `origin`, the grid index of the optical axis.
   - New methods:
     - `pad(U)` embeds the SLM field in the centre of an n × n array of zeros.
     - `crop(V)` cuts the SLM region back out.
   - `forward()` now pads before propagating. It maps `(L, n_in, n_in)` →
     `(L, n, n)`.
   - `adjoint()` now crops after back-propagating. It maps `(L, n, n)` →
     `(L, n_in, n_in)`. It is still the exact adjoint (dot test ≈ 1e-15).
   - `H` is now multiplied by a band-limit mask (Matsushima & Shimobaba 2009,
     *Opt. Express* 17, 19662). It removes the frequencies where H's phase
     changes faster than the frequency grid can sample, so z can be changed
     without the kernel aliasing.
     - With the current settings the mask removes nothing.
     - It only acts if you make z larger or the grid smaller.
2. **Cell 1, new function `choose_prop_grid(...)`:** returns the smallest
   power-of-two grid that satisfies all of the following:
   - (a) It contains the SLM and the camera, including any shift and rotation.
   - (b) It is wide enough that light diffracted off the SLM edge (up to
     ±z·tan(asin(λ/2dx)) ≈ ±1.76 mm) cannot wrap onto the camera.
   - (c) It meets N ≥ λz/dx².

   The result is multiplied by a 2× safety margin. For the current parameters
   it gives **256 × 256 (9.2 mm)**.
3. **Cell 1, `SamplingOperator.__init__`:**
   - New `origin` argument. Once the SLM is padded, the optical axis is at
     `Propagator.origin` rather than `n_prop/2`. That is not a whole number
     when n_slm is odd, as with 45.
   - It now prints a **warning** when any camera pixel falls outside the grid,
     instead of silently reading 0.
4. **Cell 2:** computes `n_prop = choose_prop_grid(...)` and passes it, plus
   `origin=prop.origin`, when building `prop`, `prop_ideal`, `sampler` and
   `sampler_ideal`. It prints the SLM, camera and grid sizes.
5. **Cells 6–7:** the visualisation now draws the real (padded) grid and the
   SLM aperture (green). Before, it labelled the SLM-sized box "Propagation
   grid".

`forward()`, `adjoint()`, `y_i()`, `spectral_init()` and `Wirtinger_iteration()`
did not need to change. The unknown x and the masks D still live on the
45 × 45 SLM grid, and padding and cropping happen inside `Propagator`.

**Check.** I compared the camera-plane field for one mask against a 1024 × 1024
reference:

| n_prop | Error vs reference |
|---|---|
| 45 (v7.1) | 85 % |
| 64 | 52 % |
| 128 | 3.3 % |
| **256 (chosen)** | **0.7 %** |
| 512 | 0.1 % |

## Fix 2: `plot2D` global-phase sign

**Problem.** `theta = angle(vdot(X_rec, X_true))` is the phase that `X_rec`
must be rotated *by* to match `X_true`. The code multiplied by `exp(-1j*theta)`,
which doubled the offset instead of removing it. That is why the recovered
phase plot looked shifted. This bug was also in v6.2.

**Change.** Cell 1, `plot2D`: `X_rec * np.exp(-1j * theta)` →
`X_rec * np.exp(1j * theta)`.

**Check.** I gave a test field a 0.7 rad offset. The old code leaves 1.4 rad;
the new code leaves 0.

## Fix 3: one geometric convention for the detector misalignment

**Problem.** `SamplingOperator` made each detector pixel read the field at
R(−θ)(x_d − s). `visualize_sampling_2d` drew it at R(θ)x_d + s. The plot
therefore showed a different misalignment from the one being simulated: the
shift sign was flipped and the rotation reversed.

**Change.** Cell 1, `SamplingOperator.__init__`: it now uses the forward map

```
Xp = ct*Xd - st*Yd + shift_x
Yp = st*Xd + ct*Yd + shift_y
```

This matches the visualiser. In physical terms, `shift_x`/`shift_y` is the
displacement of the detector centre from the optical axis, and
`rotation_deg` is the counter-clockwise rotation (x towards y) of the detector
about its own centre. The docstring now says this. `visualize_sampling_2d` was
already written in this convention. Its docstring now says so, and it takes
the same `origin`.

**Check.**
- I put a bright point in the field at x = +200 µm.
- A detector with `shift_x = +200 µm` sees it at its own centre (local x = 0).
- That is the expected result when the detector moves onto the spot.

**Note.** Any shift or rotation values you used with v7.1 now mean the
opposite sign.

## Fix 4: reconstruction uses the *ideal* model

**Problem.** The reconstruction cell called `reconstruct_x(y, D, prop, ...)`.
`prop_ideal` was never used, so a z or λ error put into `prop` was used both
to generate the data and to invert it, and so had no effect.

**Changes.**
- **Cell 5:** `reconstruct_x(y, D, prop_ideal, sampler_ideal, ...)`.
- **Cell 2:** the "true" system now has its own named parameters,
  `wavelength_true`, `z_true`, `shift_x_true`, `shift_y_true` and
  `rotation_true`. `prop` and `sampler` are built from them.
  `prop_ideal` and `sampler_ideal` use the nominal `wavelength`, `z` and zero
  misalignment. To simulate an error, change a `*_true` value; nothing else
  needs touching.
  - The grid size accounts for the largest of the true and ideal values.
- **Cell 7:** the visualisation shows the *true* detector position.

## Fix 5: reproducible random numbers

**Problem.** The masks, noise and spectral-initialisation start vector all
used the global `np.random` state with no seed. Each run gave a different
result, so you couldn't tell whether a change helped or you had just got a
lucky draw.

**Changes.**
- **Cell 2:** new `seed = 0`.
- **Cell 1:** `define_masks`, `add_noise` and `spectral_init` take an `rng`
  argument (a `np.random.Generator`).
  - Inside them, `np.random.randint` → `rng.integers`, `np.random.poisson` →
    `rng.poisson`, `np.random.normal` → `rng.normal`, `np.random.randn` →
    `rng.standard_normal`.
  - `generate_measurement` and `reconstruct_x` take `rng` and pass it on.
- **Cell 4:** `generate_measurement(..., np.random.default_rng(seed))`.
- **Cell 5:** each run uses `rng_init = np.random.default_rng(seed + 1 + i)`.
  - Re-running a cell gives the same answer.
  - Increasing `runs` gives different but repeatable initialisations.

**Check.** Two runs with the same seed give identical results.

---

## What this means for your results (please read)

With the corrected physics, **the default setup (L = 8 masks) no longer
reconstructs reliably.** The executed notebook reports a cosine accuracy of
0.35. This is not a bug in the fixes:

- Only ~19 % of the light leaving the SLM reaches the 1.92 mm camera. In v7.1
  the wrap-around folded nearly all of it back in.
- Even v7.1's own physics was fragile. With other seeds, L = 8 also failed
  there (relative error 0.99). The 0.98 you saw was partly a lucky random draw,
  which fix 5 now makes visible.

Relative error over 5 seeds, with the corrected physics and everything else at
the defaults:

| Masks L | Seed 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 8 (default) | 1.08 | 0.97 | 0.58 | 1.20 | 1.17 |
| 16 | 0.09 | 0.10 | 0.08 | 0.25 | 0.17 |
| 32 | 0.025 (single run, seed 0; the 5-seed sweep ran out of time) | | | | |

A relative error near 1 means the reconstruction failed.

What restores reconstruction: more masks, a larger camera relative to the
diffraction spread, or both. A test with L = 8 and a 3.84 mm camera gave a
13.8 % relative error.

## Compatibility fix (after first run on Colab)

- **Cell 1, imports:** `pyfftw` is now optional. If it is not installed (as on
  Colab), the code falls back to `numpy.fft` and prints a note. Results are
  identical; it is just slower (about 80 s instead of 30 s for the default run).
  To get the fast version, run `!pip install pyfftw` first.
- **Cell 6:** removed the unused `shapely` import.

## Not changed (outside fixes 1–5)

These issues from the review are still open:
- the bilinear point-sampling detector model
- the unscaled spectral initialisation and un-normalised WF step
- the cosine "accuracy" printed with a "(%)" label
- noise is still commented out
- `reconstruct_x` plots the global `x`
- `np.add.at` is slow
- `math`, `map_coordinates`, `Polygon` and `shapely` are imported but unused
- λ = 631.8 nm (check whether you meant 632.8 nm)
