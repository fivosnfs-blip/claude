"""Forward model: SLM field -> camera-plane field and intensity.

Model of the SLM field
----------------------
The SLM is a grid of square cells (pitch p). Inside a cell the field is constant
(sampled at the cell centre); with `oversample = s` every cell is split into s x s
sub-cells of pitch q = p / s. The continuous Fourier transform of such a
piecewise-constant field is exactly

    E(f) = S_q(f) * sum_n t_n exp(-2 pi i f . x_n),   S_q(f) = q^2 sinc(q fx) sinc(q fy),

i.e. a DTFT (periodic in 1/q: the higher diffraction orders) times the pixel-aperture
envelope. With `pixel_gaps` the modulated part of each pixel is a square of side
a = p sqrt(fill factor) and the gap reflects the beam unmodulated:

    E = S_a * DTFT{u e^{i phi}} + (S_p - S_a) * DTFT{u}.

Mapping to the camera
---------------------
A camera pixel at (y_c, x_c) relative to the optical axis sees the spatial frequency
f = x_c / (lambda B). B = f for the ideal set-up; it absorbs the focal-length error and
camera defocus (ABCD matrix of SLM -> d -> lens -> f + dz). The frequency step per
camera pixel is cam_pitch / (lambda B), computed here, never hard-coded. Since this step
(54.5 /m) is not 1 / (N p) for any integer N, the DTFT is evaluated directly at the
camera frequencies with a separable matrix DFT (exact resampling, no interpolation):

    E = Ey @ t @ Ex      (Ey: n_y x R_y, Ex: R_x x n_x).

Optional physics, all off by default:
* camera defocus: quadratic phase exp(i pi A r^2 / (lambda B)) on the SLM field, A = -dz/f;
* tilted plane: the SLM is tilted by theta about y, so sample x sits at x cos(theta)
  transversally and x sin(theta) deeper along the axis. The extra Fresnel propagation
  exp(-i pi lambda x sin(theta) (fx^2 + fy^2)) is applied per column (it stays separable);
* lens clipping: the spectrum is computed on a 2x finer frequency grid, propagated by d
  with the exact angular-spectrum kernel to the lens, multiplied by the circular pupil
  (and the defocus phase), and transformed to the camera again;
* pixel integration: the intensity is averaged over the 3.45 um pixel (sinc MTF).

Camera intensity is returned as power per camera pixel: |E|^2 * (cam_pitch / (lambda B))^2,
so that by Parseval sum(I) over the whole plane equals sum |u|^2 p^2 on the SLM.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.fft as sfft

from . import slm as slm_mod
from .config import SimConfig


def _sinc_env(width: float, freqs: np.ndarray) -> np.ndarray:
    """1-D Fourier transform of a rect of the given width."""
    return width * np.sinc(width * freqs)


class ForwardModel:
    def __init__(self, cfg: SimConfig, field_fn):
        """field_fn(Y, X) -> complex field on the SLM surface (metres from the optical axis)."""
        self.cfg = cfg
        cam = cfg.camera
        lam = cfg.wavelength
        p = cfg.slm_pitch
        s = int(cfg.oversample)
        if s < 1:
            raise ValueError("oversample must be >= 1")
        self.s = s
        self.ctype = np.dtype(cfg.dtype)
        rt = np.float32 if self.ctype == np.complex64 else np.float64
        My, Mx = cfg.slm_shape
        ay, ax = cfg.slm_axis()

        # --- geometry of the Fourier transform ------------------------------------
        self.f_true = cfg.f * (1 + cfg.focal_error) if cfg.focal_length_error else cfg.f
        self.dz = cfg.defocus_dz if cfg.camera_defocus else 0.0
        if cfg.lens_clipping:
            # Defocus is applied in the lens plane; scale set by lens -> camera distance.
            self.B = self.f_true + self.dz
            self.A = 0.0
        else:
            self.B = self.f_true + self.dz * (1.0 - cfg.d_slm_lens / self.f_true)
            self.A = -self.dz / self.f_true
        self.freq_step = cam.pitch / (lam * self.B)            # true, used to simulate
        self.freq_step_nominal = cam.pitch / (lam * cfg.f)     # what Part 2 is told
        theta = math.radians(cfg.theta_deg) if cfg.tilted_plane else 0.0
        self.cos_t, self.sin_t = math.cos(theta), math.sin(theta)

        # --- the field on the native pixel grid and the crop ------------------------
        yv = (np.arange(My) - ay) * p
        xv = (np.arange(Mx) - ax) * p
        u_full = np.asarray(field_fn(yv[:, None], xv[None, :]), dtype=np.complex128)
        u_full = np.broadcast_to(u_full, (My, Mx)).copy()
        self.u_true = u_full
        amp = np.abs(u_full)
        if not np.any(amp > 0):
            raise ValueError("beam is zero on the SLM")
        rows = np.nonzero(np.any(amp > cfg.crop_threshold * amp.max(), axis=1))[0]
        cols = np.nonzero(np.any(amp > cfg.crop_threshold * amp.max(), axis=0))[0]
        y0, y1 = max(rows[0] - 1, 0), min(rows[-1] + 2, My)
        x0, x1 = max(cols[0] - 1, 0), min(cols[-1] + 2, Mx)
        self.crop = tuple(int(v) for v in (y0, y1, x0, x1))
        self.u_nat = u_full[y0:y1, x0:x1]
        self._check_phase_sampling()

        ys_n, xs_n = yv[y0:y1], xv[x0:x1]
        if s == 1:
            ys, xs = ys_n, xs_n
            self.u_sub = self.u_nat
        else:
            ys = (y0 + (np.arange((y1 - y0) * s) + 0.5) / s - 0.5 - ay) * p
            xs = (x0 + (np.arange((x1 - x0) * s) + 0.5) / s - 0.5 - ax) * p
            self.u_sub = np.asarray(field_fn(ys[:, None], xs[None, :]), dtype=np.complex128)

        # --- camera frequencies --------------------------------------------------------
        ry0, rx0, rny, rnx = cfg.roi_box()
        cay, cax = cam.axis()
        self.mg_pix = 16 if cfg.pixel_integration else 0
        self.mg_clip = 8 if cfg.lens_clipping else 0
        mg = self.mg_pix + self.mg_clip
        cam_rows = np.arange(ry0 - mg, ry0 + rny + mg)
        cam_cols = np.arange(rx0 - mg, rx0 + rnx + mg)
        fy = (cam_rows - cay) * self.freq_step
        fx = (cam_cols - cax) * self.freq_step
        if cfg.lens_clipping:
            fy = fy[0] + np.arange(2 * fy.size) * (self.freq_step / 2)
            fx = fx[0] + np.arange(2 * fx.size) * (self.freq_step / 2)
        self.fy, self.fx = fy, fx

        # --- transforms and envelopes ------------------------------------------------
        q = p / s
        self.M_sub = self._matrices(ys, xs)
        ey_q, ex_q = _sinc_env(q, fy), _sinc_env(q * self.cos_t, fx)
        self.gaps = bool(cfg.pixel_gaps) and cfg.fill_factor < 1.0
        if self.gaps:
            a = p * math.sqrt(cfg.fill_factor)
            Sp = np.outer(_sinc_env(p, fy), _sinc_env(p * self.cos_t, fx))
            Sa = np.outer(_sinc_env(a, fy), _sinc_env(a * self.cos_t, fx))
            self.S_gap = (Sp - Sa).astype(rt)
            self.M_nat = self.M_sub if s == 1 else self._matrices(ys_n, xs_n)
            self.gap_const = self.S_gap * self._mft(self.u_nat, self.M_nat)
            if s == 1:
                self.env = (Sa.astype(rt), None)
            else:
                self.env = (ey_q.astype(rt), ex_q.astype(rt))
        else:
            self.env = (ey_q.astype(rt), ex_q.astype(rt))

        if cfg.lens_clipping:
            self._setup_clipping()

    # ------------------------------------------------------------------------------
    def _check_phase_sampling(self):
        """Warn if the beam phase is undersampled by the SLM pixels."""
        u = self.u_nat
        a = np.abs(u)
        m = a > 1e-3 * a.max()
        dpx = np.abs(np.angle(u[:, 1:] * np.conj(u[:, :-1])))[m[:, 1:] & m[:, :-1]]
        dpy = np.abs(np.angle(u[1:, :] * np.conj(u[:-1, :])))[m[1:, :] & m[:-1, :]]
        worst = max(dpx.max(initial=0), dpy.max(initial=0))
        self.max_phase_step = float(worst)
        if worst > math.pi / 2:
            import warnings
            warnings.warn(f"beam phase changes by {worst:.2f} rad between SLM pixels; "
                          "the field is undersampled")

    def _matrices(self, ys: np.ndarray, xs: np.ndarray) -> dict:
        cfg = self.cfg
        lam = cfg.wavelength
        fy, fx = self.fy, self.fx
        ct = self.ctype
        Ey = np.exp(-2j * np.pi * np.outer(fy, ys)).astype(ct)
        ph_x = -2 * np.pi * np.outer(xs, fx) * self.cos_t
        T = None
        if self.sin_t:
            # sample at x is a distance x sin(theta) further from the lens (paraxial ASM)
            ph_x = ph_x - np.pi * lam * self.sin_t * np.outer(xs, fx**2)
            T = np.exp(-1j * np.pi * lam * self.sin_t * np.outer(fy**2, xs)).astype(ct)
        Ex = np.exp(1j * ph_x).astype(ct)
        Q = None
        if self.A:
            r2 = (xs[None, :] * self.cos_t) ** 2 + ys[:, None] ** 2
            Q = np.exp(1j * np.pi * self.A * r2 / (lam * self.B))
        return {"Ey": Ey, "Ex": Ex, "T": T, "Q": Q}

    def _mft(self, t: np.ndarray, M: dict) -> np.ndarray:
        if M["Q"] is not None:
            t = t * M["Q"]
        G = M["Ey"] @ np.asarray(t, dtype=self.ctype)
        if M["T"] is not None:
            G *= M["T"]
        return G @ M["Ex"]

    def _setup_clipping(self):
        cfg = self.cfg
        lam = cfg.wavelength
        Ly, Lx = self.fy.size, self.fx.size
        FY, FX = self.fy[:, None], self.fx[None, :]
        f2 = FX**2 + FY**2
        # exp(2 pi i d (sqrt(1/lam^2 - f^2) - 1/lam)), written in a cancellation-free form
        kz_minus_k = -lam * f2 / (1.0 + np.sqrt(np.maximum(1.0 - lam**2 * f2, 0.0)))
        H = np.exp(2j * np.pi * cfg.d_slm_lens * kz_minus_k)
        H[lam**2 * f2 >= 1.0] = 0.0
        self.H_d = H.astype(self.ctype)
        dfy, dfx = self.fy[1] - self.fy[0], self.fx[1] - self.fx[0]
        yL = np.fft.fftfreq(Ly, d=dfy)[:, None]          # lens-plane coordinates
        xL = np.fft.fftfreq(Lx, d=dfx)[None, :]
        pupil = ((xL - cfg.lens_decenter) ** 2 + yL**2) <= cfg.lens_clear_radius**2
        lensQ = np.ones((Ly, Lx), dtype=np.complex128)
        if self.dz:
            lensQ = np.exp(1j * np.pi * (xL**2 + yL**2)
                           * (1.0 / (self.f_true + self.dz) - 1.0 / self.f_true) / lam)
        self.lens_mult = (pupil * lensQ).astype(self.ctype)
        self.lens_span = (1.0 / dfy, 1.0 / dfx)

    def _clip(self, E: np.ndarray) -> np.ndarray:
        EL = sfft.ifft2(E * self.H_d, workers=-1)
        E2 = sfft.fft2(EL * self.lens_mult, workers=-1)[::2, ::2]
        m = self.mg_clip
        return E2[m:E2.shape[0] - m, m:E2.shape[1] - m]

    # ------------------------------------------------------------------------------
    def applied(self, phi_cmd: np.ndarray):
        """Applied sub-grid phase on the crop, from the commanded full-SLM phase."""
        cfg = self.cfg
        y0, y1, x0, x1 = self.crop
        My, Mx = cfg.slm_shape
        m = slm_mod.walkoff_margin(cfg)
        wy0, wy1, wx0, wx1 = max(y0 - m, 0), min(y1 + m, My), max(x0 - m, 0), min(x1 + m, Mx)
        phi = slm_mod.applied_phase(np.asarray(phi_cmd)[wy0:wy1, wx0:wx1], cfg)
        s = self.s
        oy, ox = (y0 - wy0) * s, (x0 - wx0) * s
        return phi[oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s]

    def field(self, phi_cmd: np.ndarray | None = None) -> np.ndarray:
        """Complex camera-plane spectrum E(f) (before |.|^2) on the ROI (+ integration margin).

        phi_cmd: commanded (My, Mx) phase in radians; None means a flat mask."""
        if phi_cmd is None:
            phi_cmd = np.zeros(self.cfg.slm_shape)
        phi_sub = self.applied(phi_cmd)
        E = self._mft(self.u_sub * np.exp(1j * phi_sub), self.M_sub)
        ey, ex = self.env
        if ex is None:
            E *= ey
        else:
            E *= ey[:, None]
            E *= ex[None, :]
        if self.gaps:
            if self.s == 1:
                E += self.gap_const
            else:
                ebar = slm_mod.pixel_mean_field(phi_sub, self.s)
                E += self.gap_const - self.S_gap * self._mft(self.u_nat * ebar, self.M_nat)
        if self.cfg.lens_clipping:
            E = self._clip(E)
        return E

    def intensity(self, phi_cmd: np.ndarray | None = None) -> np.ndarray:
        """Noise-free power per camera pixel on the ROI, float64."""
        E = self.field(phi_cmd)
        I = (E.real.astype(np.float64) ** 2 + E.imag.astype(np.float64) ** 2) * self.freq_step**2
        if self.cfg.pixel_integration:
            ny, nx = I.shape
            mtf = np.outer(np.sinc(np.fft.fftfreq(ny)), np.sinc(np.fft.fftfreq(nx)))
            I = np.maximum(sfft.ifft2(sfft.fft2(I, workers=-1) * mtf, workers=-1).real, 0.0)
            m = self.mg_pix
            I = I[m:ny - m, m:nx - m]
        return I

    @property
    def slm_power(self) -> float:
        p = self.cfg.slm_pitch
        return float(np.sum(np.abs(self.u_true) ** 2) * p * p)

    def camera_coords(self) -> tuple:
        """(y, x) camera-plane coordinates in metres of the ROI pixels, from the axis."""
        cam = self.cfg.camera
        ry0, rx0, rny, rnx = self.cfg.roi_box()
        cay, cax = cam.axis()
        return ((np.arange(ry0, ry0 + rny) - cay) * cam.pitch,
                (np.arange(rx0, rx0 + rnx) - cax) * cam.pitch)
