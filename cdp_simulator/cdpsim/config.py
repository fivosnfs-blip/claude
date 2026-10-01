"""Configuration of the simulated set-up.

All lengths are in metres, angles in degrees, phases in radians.
Every non-ideal effect is a boolean switch that is off by default; its
magnitude is a separate parameter that only matters when the switch is on.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass
class BeamConfig:
    """Incoming HeNe beam, defined on the SLM surface (the reconstruction plane)."""

    diameter: float = 4e-3            # 1/e^2 intensity diameter (max ~8 mm)
    center: tuple | None = None       # (y, x) in SLM pixel coordinates; None -> optical axis
    power: float = 1.0                # total power, sum |u|^2 * pitch^2 (arbitrary units)
    norm_radius: float = 3e-3         # radius that maps to rho = 1 in the Zernike terms
    # Zernike coefficients in radians (orthonormal over the unit disc).
    zernike: dict = field(default_factory=lambda: {
        "tilt_x": 1.0, "tilt_y": -0.6, "defocus": 1.2,
        "astig_0": 0.7, "astig_45": -0.4, "coma_x": 0.25,
    })
    vortex_charge: int = 0            # l != 0 adds exp(i l theta) and an r^|l| core (LG_0l)

    @property
    def extent(self) -> float:
        """Clear field extent D_ext = 1.5 D used for the sampling budget."""
        return 1.5 * self.diameter


@dataclass
class CameraConfig:
    pitch: float = 3.45e-6
    shape: tuple = (2048, 2448)       # (Ny, Nx)
    bits: int = 12
    qe: float = 0.65                  # quantum efficiency at 633 nm
    full_well: float = 10500.0        # electrons
    gain: float | None = None         # ADU per electron; None -> full well maps to full scale
    offset: float = 32.0              # dark offset, ADU
    read_noise: float = 2.5           # electrons rms
    axis_pixel: tuple | None = None   # (y, x) pixel hit by the optical axis; None -> (Ny//2, Nx//2)
    # Exposure: 'global' = one exposure for all K images, peak of the brightest image
    # at peak_fraction of full scale; 'per_image' = each image scaled on its own;
    # 'fixed' = use photons_per_unit as given.
    exposure: str = "global"
    peak_fraction: float = 0.8
    photons_per_unit: float | None = None
    shot_noise: bool = True
    quantize: bool = True

    @property
    def full_scale(self) -> int:
        return 2 ** self.bits - 1

    @property
    def adu_per_e(self) -> float:
        return self.gain if self.gain is not None else self.full_scale / self.full_well

    def axis(self) -> tuple:
        if self.axis_pixel is not None:
            return tuple(float(v) for v in self.axis_pixel)
        return (float(self.shape[0] // 2), float(self.shape[1] // 2))


@dataclass
class SimConfig:
    # Source and Fourier optics
    wavelength: float = 632.8e-9
    f: float = 0.100                  # nominal focal length (what Part 2 is told)
    d_slm_lens: float = 0.110         # SLM -> lens optical path
    lens_clear_radius: float = 11e-3
    lens_decenter: float = 0.0        # lens axis offset in x at the lens plane

    # SLM (Hamamatsu X15213-01)
    slm_pitch: float = 12.5e-6
    slm_shape: tuple = (1024, 1272)   # (My, Mx)
    slm_bits: int = 8
    slm_axis_pixel: tuple | None = None  # (y, x) where the optical axis meets the SLM
    theta_deg: float = 5.0            # incidence angle (only used by tilted_plane)

    # Masks
    n_masks: int = 8
    mask_block: int = 1               # 1 -> 1x1 pixel blocks, 2 -> 2x2
    seed: int = 0

    # Numerics
    oversample: int = 1               # sub-samples per SLM pixel and axis (1, 2, 4, ...)
    crop_threshold: float = 1e-4      # beam amplitude (rel. to max) below which the SLM is ignored
    dtype: str = "complex64"
    roi: tuple | None = None          # (y0, x0, ny, nx) on the camera; None -> full sensor

    # ---- Non-ideal effects: all switches off by default -----------------------------
    pixel_gaps: bool = False          # finite fill factor: gaps reflect unmodulated light
    fill_factor: float = 0.968
    walkoff: bool = False             # LC walk-off blur along x (double pass, tilt)
    walkoff_dx: float = 0.55e-6
    tilted_plane: bool = False        # position-dependent propagation distance x*sin(theta)
    phase_nonlinearity: bool = False  # applied = range * (commanded/2pi)^gamma
    phase_range: float = 0.92 * 2 * math.pi
    phase_gamma: float = 1.15
    crosstalk: bool = False           # fringing-field blur of the phase pattern
    crosstalk_sigma_px: float = 0.42  # Gaussian sigma in SLM pixels (FWHM ~ 1 pixel)
    lens_clipping: bool = False       # finite lens aperture (needs angular-spectrum step)
    camera_defocus: bool = False
    defocus_dz: float = 0.2e-3        # camera displacement from the back focal plane
    focal_length_error: bool = False
    focal_error: float = 0.01         # relative error, f_true = f * (1 + focal_error)
    pixel_integration: bool = False   # camera pixel integrates intensity over its area

    camera: CameraConfig = field(default_factory=CameraConfig)

    def slm_axis(self) -> tuple:
        if self.slm_axis_pixel is not None:
            return tuple(float(v) for v in self.slm_axis_pixel)
        My, Mx = self.slm_shape
        return ((My - 1) / 2.0, (Mx - 1) / 2.0)

    def roi_box(self) -> tuple:
        if self.roi is not None:
            return tuple(int(v) for v in self.roi)
        Ny, Nx = self.camera.shape
        return (0, 0, int(Ny), int(Nx))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        d = dict(d)
        cam = CameraConfig(**_tuples(d.pop("camera", {})))
        return cls(camera=cam, **_tuples(d))


def beam_from_dict(d: dict) -> BeamConfig:
    return BeamConfig(**_tuples(d))


def _tuples(d: dict) -> dict:
    """JSON turns tuples into lists; turn them back."""
    return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}


def sampling_budget(cfg: SimConfig, beam: BeamConfig) -> dict:
    """Design numbers from the layout: frequency step, oversampling, spectrum width."""
    lam, f = cfg.wavelength, cfg.f
    cam = cfg.camera
    freq_step = cam.pitch / (lam * f)                 # cycles/m per camera pixel
    d_ext = beam.extent
    sigma = lam * f / (cam.pitch * d_ext)
    spectrum = lam * f / (cfg.mask_block * cfg.slm_pitch)
    sensor_h = cam.shape[0] * cam.pitch
    sensor_w = cam.shape[1] * cam.pitch
    return {
        "freq_step_per_m": freq_step,
        "sigma_oversampling": sigma,
        "D_ext_m": d_ext,
        "spectrum_width_m": spectrum,
        "spectrum_width_px": spectrum / cam.pitch,
        "sensor_height_m": sensor_h,
        "sensor_width_m": sensor_w,
        "spectrum_fits_sensor": spectrum <= min(sensor_h, sensor_w),
        "speckle_size_m": lam * f / d_ext,
    }
