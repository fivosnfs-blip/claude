"""Physics and data-contract tests for the Part 1 simulator.

Run from cdp_simulator/:  python -m pytest -q
"""
import json
import math

import numpy as np
import pytest

from cdpsim import Beam, BeamConfig, ForwardModel, SimConfig
from cdpsim.camera import expose
from cdpsim.config import CameraConfig, sampling_budget
from cdpsim.simulate import simulate
from cdpsim.slm import applied_phase, generate_masks

LAM, F, P = 632.8e-9, 0.100, 12.5e-6
W = 2e-3  # 1/e^2 amplitude radius of the 4 mm beam


def gaussian(w=W, y0=0.0, x0=0.0, phase=None):
    def fn(Y, X):
        u = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / w**2) + 0j
        return u * np.exp(1j * phase(Y, X)) if phase else u
    return fn


def moments(model, I, deenvelope=False):
    yc, xc = model.camera_coords()
    if deenvelope:  # remove the pixel-aperture sinc^2, which slightly pulls the centroid inwards
        I = I / (np.sinc(P * yc / (LAM * F))[:, None] * np.sinc(P * xc / (LAM * F))[None, :]) ** 2
    tot = I.sum()
    my = (I.sum(1) * yc).sum() / tot
    mx = (I.sum(0) * xc).sum() / tot
    vy = (I.sum(1) * (yc - my) ** 2).sum() / tot
    vx = (I.sum(0) * (xc - mx) ** 2).sum() / tot
    return my, mx, vy, vx


def cfg128(**kw):
    return SimConfig(dtype="complex128", crop_threshold=1e-6, **kw)


# --- geometry -------------------------------------------------------------------------

def test_frequency_step_is_computed_from_parameters():
    m = ForwardModel(SimConfig(), gaussian())
    assert m.freq_step == pytest.approx(3.45e-6 / (LAM * F), rel=1e-12)
    m2 = ForwardModel(SimConfig(f=0.085), gaussian())
    assert m2.freq_step == pytest.approx(3.45e-6 / (LAM * 0.085), rel=1e-12)
    b = sampling_budget(SimConfig(), BeamConfig())
    assert b["sigma_oversampling"] == pytest.approx(3.057, abs=1e-3)
    assert b["spectrum_width_m"] == pytest.approx(5.062e-3, abs=1e-6)
    assert b["spectrum_fits_sensor"]


def test_matrix_dft_matches_direct_sum():
    """The separable matrix DFT equals the brute-force DTFT at arbitrary camera pixels."""
    cfg = SimConfig(dtype="complex128", slm_shape=(64, 80), crop_threshold=0.0)
    model = ForwardModel(cfg, gaussian(w=0.25e-3, phase=lambda Y, X: 3e3 * X - 4e6 * Y**2))
    phi = generate_masks(SimConfig(slm_shape=(64, 80), n_masks=1), np.random.default_rng(1))[0]
    E = model.field(phi)
    ay, ax = cfg.slm_axis()
    yy = (np.arange(64) - ay)[:, None] * P
    xx = (np.arange(80) - ax)[None, :] * P
    t = model.u_true * np.exp(1j * phi.astype(np.float64))
    rng = np.random.default_rng(2)
    for _ in range(20):
        i, j = rng.integers(0, 2048), rng.integers(0, 2448)
        fy, fx = model.fy[i], model.fx[j]
        ref = np.sum(t * np.exp(-2j * np.pi * (fy * yy + fx * xx)))
        ref *= P * np.sinc(P * fy) * P * np.sinc(P * fx)
        assert abs(E[i, j] - ref) <= 1e-10 * np.abs(E).max()


# --- the three required physics tests ---------------------------------------------------

def test_flat_mask_gives_fourier_transform_of_beam_only():
    beam = Beam(BeamConfig(zernike={}), cfg128())
    model = ForwardModel(cfg128(), beam)
    I = model.intensity(None)
    yc, xc = model.camera_coords()
    fy, fx = yc / (LAM * F), xc / (LAM * F)
    # continuous FT of exp(-r^2/w^2) is pi w^2 exp(-pi^2 w^2 f^2); pixel envelope sinc^2
    E_ref = beam.scale * math.pi * W**2 * np.exp(-math.pi**2 * W**2 * (fy[:, None] ** 2 + fx[None, :] ** 2))
    I_ref = np.abs(E_ref) ** 2 * (np.sinc(P * fy)[:, None] * np.sinc(P * fx)[None, :]) ** 2 * model.freq_step**2
    # 1e-5: the 12.8 mm SLM height truncates the Gaussian at 3.6e-5 of its peak amplitude
    assert np.abs(I - I_ref).max() < 3e-5 * I_ref.max()
    # nothing else: no light beyond 6 spot radii, and all the beam power is on the sensor
    W_cam = LAM * F / (math.pi * W)
    r = np.hypot(yc[:, None], xc[None, :])
    assert I[r > 6 * W_cam].sum() < 1e-10 * I.sum()
    # Parseval: everything except the p^2/(6 w^2) = 6.5e-6 that the pixel sinc^2 sends to
    # the higher orders (in the envelope's zeros, off the sensor) lands on the sensor
    assert I.sum() == pytest.approx(model.slm_power * (1 - P**2 / (6 * W**2)), rel=2e-7)


def test_quadratic_phase_spreads_spot_as_predicted():
    R = 10.0  # radius of curvature of the wavefront on the SLM
    quad = lambda Y, X: -math.pi * (X**2 + Y**2) / (LAM * R)
    m0 = ForwardModel(cfg128(), gaussian())
    m1 = ForwardModel(cfg128(), gaussian(phase=quad))
    _, _, vy0, vx0 = moments(m0, m0.intensity(None))
    my, mx, vy1, vx1 = moments(m1, m1.intensity(None))
    # Gaussian optics: camera 1/e^2 radius W = (lam f / pi w) sqrt(1 + (pi w^2 / lam R)^2), <x^2> = W^2/4
    W0 = LAM * F / (math.pi * W)
    W1 = W0 * math.sqrt(1 + (math.pi * W**2 / (LAM * R)) ** 2)
    assert math.sqrt(vx0) == pytest.approx(W0 / 2, rel=1e-4)
    assert math.sqrt(vx1) == pytest.approx(W1 / 2, rel=1e-4)
    assert math.sqrt(vy1) == pytest.approx(W1 / 2, rel=1e-4)
    assert abs(mx) < 1e-9 and abs(my) < 1e-9


def test_quadratic_phase_on_offset_beam_shifts_spot_as_predicted():
    """A curved wavefront on a beam sitting x0 off axis tilts it locally: centroid at -f x0 / R."""
    R, x0, y0 = 2.0, 1.0e-3, -0.5e-3
    quad = lambda Y, X: -math.pi * (X**2 + Y**2) / (LAM * R)
    m = ForwardModel(cfg128(), gaussian(x0=x0, y0=y0, phase=quad))
    my, mx, _, _ = moments(m, m.intensity(None), deenvelope=True)
    assert mx == pytest.approx(-F * x0 / R, rel=2e-4)
    assert my == pytest.approx(-F * y0 / R, rel=2e-4)


def test_linear_phase_shifts_spot_by_lambda_f_a():
    a = 300.0  # cycles per metre on the SLM
    m = ForwardModel(cfg128(), gaussian(phase=lambda Y, X: 2 * math.pi * a * X))
    _, mx, _, _ = moments(m, m.intensity(None))
    assert mx == pytest.approx(LAM * F * a, rel=1e-4)


def test_doubling_field_quadruples_intensity():
    cfg = SimConfig()
    beam = Beam(BeamConfig(), cfg)
    phi = generate_masks(cfg, np.random.default_rng(3))[0]
    I1 = ForwardModel(cfg, beam).intensity(phi)
    I2 = ForwardModel(cfg, lambda Y, X: 2 * beam(Y, X)).intensity(phi)
    np.testing.assert_allclose(I2, 4 * I1, rtol=1e-5, atol=1e-6 * I1.max())
    # through the camera with a fixed exposure: offset-subtracted mean scales by 4
    cam = CameraConfig(exposure="fixed")
    ppu = 0.2 * cam.full_well / (cam.qe * I2.max())
    rng = np.random.default_rng(4)
    a1, _ = expose(I1, ppu, cam, rng)
    a2, _ = expose(I2, ppu, cam, rng)
    assert (a2.mean() - cam.offset) / (a1.mean() - cam.offset) == pytest.approx(4.0, rel=2e-3)


# --- power conservation --------------------------------------------------------------

def test_parseval_random_mask_matches_envelope_prediction():
    cfg = SimConfig()
    model = ForwardModel(cfg, Beam(BeamConfig(), cfg))
    phi = generate_masks(cfg, np.random.default_rng(5))[0]
    frac = model.intensity(phi).sum() / model.slm_power
    # random independent phases: E|E(f)|^2 = |S_p(f)|^2 sum |u|^2  ->  envelope integral
    yc, xc = model.camera_coords()
    fy, fx = yc / (LAM * F), xc / (LAM * F)
    pred = (np.sinc(P * fy) ** 2).sum() * (np.sinc(P * fx) ** 2).sum() * (P * model.freq_step) ** 2
    assert frac == pytest.approx(pred, rel=5e-3)


# --- switches -----------------------------------------------------------------------------

def test_camera_defocus_matches_gaussian_beam_optics():
    dz = 0.2e-3
    m = ForwardModel(cfg128(camera_defocus=True, defocus_dz=dz), gaussian())
    _, _, _, vx = moments(m, m.intensity(None))
    W0 = LAM * F / (math.pi * W)           # waist in the back focal plane
    zR = math.pi * W0**2 / LAM
    assert math.sqrt(vx) == pytest.approx(W0 / 2 * math.sqrt(1 + (dz / zR) ** 2), rel=2e-3)


def test_focal_length_error_rescales_pattern():
    m0 = ForwardModel(cfg128(), gaussian(phase=lambda Y, X: 2 * math.pi * 300 * X))
    m1 = ForwardModel(cfg128(focal_length_error=True, focal_error=0.01),
                      gaussian(phase=lambda Y, X: 2 * math.pi * 300 * X))
    _, mx0, _, _ = moments(m0, m0.intensity(None))
    _, mx1, _, _ = moments(m1, m1.intensity(None))
    assert mx1 / mx0 == pytest.approx(1.01, rel=1e-4)


def test_tilted_plane_foreshortens_beam():
    m0 = ForwardModel(cfg128(), gaussian())
    m1 = ForwardModel(cfg128(tilted_plane=True), gaussian())
    _, _, vy0, vx0 = moments(m0, m0.intensity(None))
    _, _, vy1, vx1 = moments(m1, m1.intensity(None))
    assert math.sqrt(vx1 / vx0) == pytest.approx(1 / math.cos(math.radians(5)), rel=1e-4)
    assert vy1 == pytest.approx(vy0, rel=1e-4)


def test_lens_clipping_path_is_neutral_for_small_beam_and_clips_large_one():
    cfg = SimConfig()
    beam = Beam(BeamConfig(), cfg)
    phi = generate_masks(cfg, np.random.default_rng(6))[0]
    I0 = ForwardModel(cfg, beam).intensity(phi)
    Ic = ForwardModel(SimConfig(lens_clipping=True), beam).intensity(phi)
    assert abs(Ic.sum() - I0.sum()) < 2e-4 * I0.sum()
    # 2 mm aperture radius, much smaller than the 10 mm diffraction bundle at the lens
    Is = ForwardModel(SimConfig(lens_clipping=True, lens_clear_radius=2e-3), beam).intensity(phi)
    assert Is.sum() < 0.6 * I0.sum()


def test_pixel_gaps_add_zero_order_light():
    cfg = SimConfig()
    beam = Beam(BeamConfig(zernike={}), cfg)
    phi = generate_masks(cfg, np.random.default_rng(7))[0]
    I0 = ForwardModel(cfg, beam).intensity(phi)
    Ig = ForwardModel(SimConfig(pixel_gaps=True, fill_factor=0.968), beam).intensity(phi)
    ay, ax = (int(v) for v in cfg.camera.axis())
    assert Ig[ay, ax] > 3 * I0[ay, ax]   # bright unmodulated spot at the axis
    # flat mask: gaps make no difference
    np.testing.assert_allclose(ForwardModel(SimConfig(pixel_gaps=True), beam).intensity(None),
                               ForwardModel(cfg, beam).intensity(None), rtol=1e-4, atol=1e-9)


@pytest.mark.parametrize("switch", ["walkoff", "crosstalk", "phase_nonlinearity"])
def test_slm_imperfections_change_applied_not_commanded(switch):
    cfg = SimConfig(oversample=4, **{switch: True})
    phi = generate_masks(SimConfig(n_masks=1), np.random.default_rng(8))[0][:40, :40]
    ideal = applied_phase(phi, SimConfig(oversample=4))
    applied = applied_phase(phi, cfg)
    assert np.abs(applied - ideal).max() > 1e-3
    np.testing.assert_array_equal(ideal[::4, ::4], np.mod(phi, 2 * np.pi))


def test_oversampling_converges_to_native_without_imperfections():
    cfg = SimConfig()
    beam = Beam(BeamConfig(), cfg)
    phi = generate_masks(cfg, np.random.default_rng(9))[0]
    I1 = ForwardModel(cfg, beam).intensity(phi)
    I2 = ForwardModel(SimConfig(oversample=2), beam).intensity(phi)
    I4 = ForwardModel(SimConfig(oversample=4), beam).intensity(phi)
    # same piecewise-constant mask; only the beam varies inside each pixel. That variation
    # is a ~0.5% effect (L1) and it converges as the sub-grid is refined.
    d14, d24 = np.abs(I4 - I1).sum() / I1.sum(), np.abs(I4 - I2).sum() / I1.sum()
    assert d14 < 1e-2 and d24 < 0.5 * d14


# --- masks and data contract -------------------------------------------------------------

def test_masks_are_quantised_and_blocked():
    phi = generate_masks(SimConfig(n_masks=2, mask_block=2), np.random.default_rng(10))
    assert phi.shape == (2, 1024, 1272) and phi.dtype == np.float32
    g = phi.astype(np.float64) / (2 * np.pi / 256)
    np.testing.assert_allclose(g, np.round(g), atol=1e-3)
    assert phi.min() >= 0 and phi.max() < 2 * np.pi
    np.testing.assert_array_equal(phi[:, 0::2, 0::2], phi[:, 1::2, 1::2])


def test_data_contract(tmp_path):
    cfg = SimConfig(n_masks=2, roi=(512, 600, 1024, 1200))
    simulate(cfg, BeamConfig(), tmp_path, verbose=False)
    d = np.load(tmp_path / "measurements.npz")
    assert set(d.files) == {"I", "phi", "meta"}
    assert d["I"].shape == (2, 1024, 1200) and d["phi"].shape == (2, 1024, 1272)
    meta = json.loads(str(d["meta"]))
    for key in ("wavelength", "f", "slm_pitch", "slm_shape", "cam_pitch", "cam_shape", "mask_block",
                "bits_cam", "gain", "offset", "read_noise", "beam_center_on_slm", "roi"):
        assert key in meta, key
    assert meta["slm_shape"] == [1024, 1272] and meta["cam_shape"] == [2048, 2448]
    assert meta["roi"] == [512, 600, 1024, 1200]
    I = d["I"]
    assert I.min() >= 0 and I.max() <= 4095 and np.all(I == np.round(I))
    t = np.load(tmp_path / "truth.npz")
    assert t["u_true"].shape == (1024, 1272) and np.iscomplexobj(t["u_true"])
    # meta must not reveal the hidden imperfections
    assert "tilted_plane" not in meta and "crosstalk" not in meta
