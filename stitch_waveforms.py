from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.signal.windows import tukey
from typing import Dict, Tuple, Union, TYPE_CHECKING

# Only imported when static type‑checkers (e.g. Pylance, mypy) are running –
# has *no* runtime cost and silences "unknown module" warnings if PyCBC is
# not in the current environment.
if TYPE_CHECKING:
    from pycbc.types import TimeSeries  # type: ignore  # noqa: F401

Array = Union[np.ndarray, "TimeSeries"]

# -----------------------------------------------------------------------------
# Helper: Window shapes (for blending)
# -----------------------------------------------------------------------------

def _blend_window(length: int, kind: str = "raised_cosine") -> np.ndarray:
    if kind == "raised_cosine":
        return 0.5 * (1 - np.cos(np.linspace(0, np.pi, length)))
    if kind == "tukey":
        return tukey(length, alpha=0.5)
    if kind == "linear":
        return np.linspace(0, 1, length)
    if kind == "hanning":
        return np.hanning(length)
    if kind == "hamming":
        return np.hamming(length)
    raise ValueError(
        "transition_type must be one of 'raised_cosine', 'tukey', 'linear', "
        "'hanning', 'hamming'."
    )

# -----------------------------------------------------------------------------
# Core stitcher for *one* real waveform (used for + and ×)
# -----------------------------------------------------------------------------

def _stitch_real(
    h1: Array,
    h2: Array,
    stitch_time: float,
    transition_length: float,
    delta_t: float,
    phase_shift: float,
    amp_scale: float,
    window_kind: str,
):
    """Return stitched real strain and common time array."""
    # Truncate to common length
    n = min(len(h1), len(h2))
    h1 = h1[:n]
    h2 = h2[:n]
    t = h1.sample_times[:n] if hasattr(h1, "sample_times") else np.arange(n) * delta_t

    # Apply same alignment derived from h+ to this polarisation
    analytic2 = hilbert(h2)
    amp2 = np.abs(analytic2)
    phase2 = np.unwrap(np.angle(analytic2))
    phase2_aligned = phase2 - phase_shift
    h2_aligned = amp2 * amp_scale * np.cos(phase2_aligned)

    # Blending indices
    half = int((transition_length / 2) / delta_t)
    idx = np.argmin(np.abs(t - stitch_time))
    i0, i1 = idx - half, idx + half
    win = _blend_window(i1 - i0 + 1, window_kind)

    # Combine
    h = np.zeros_like(h1)
    h[:i0] = h1[:i0]
    h[i0 : i1 + 1] = h1[i0 : i1 + 1] * (1 - win) + h2_aligned[i0 : i1 + 1] * win
    h[i1 + 1 :] = h2_aligned[i1 + 1 :]
    return h, t

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def stitch_event(
    hp1: Array,
    hc1: Array,
    hp2: Array,
    hc2: Array,
    *,
    detector_dict: Dict[str, Tuple[Array, Array]],
    stitch_time: float = -0.04,
    transition_length: float = 1e-3,
    transition_type: str = "raised_cosine",
    delta_t: float = 2 ** -12,
    show_plots: bool = False,
):
    """Stitch two IMR waveforms (both polarisations) and project onto detectors.

    Parameters
    ----------
    hp1, hc1 : array‑like
        + / × polarisations of *segment 1* in the source frame.
    hp2, hc2 : array‑like
        + / × polarisations of *segment 2* (post‑transition).
    detector_dict : dict
        Mapping ``ifo_name -> (Fplus, Fcross)``.  ``Fplus/Fcross`` may be
        scalars or 1‑D numpy arrays of same length as the waveforms.
    stitch_time : float
        Transition timestamp (seconds, same convention as times in waveforms).
    transition_length : float
        Width of the blending window (seconds).
    transition_type : str
        One of 'raised_cosine', 'tukey', 'linear', 'hanning', 'hamming'.
    delta_t : float
        Sample spacing (s) – only used if arrays don’t carry time info.
    show_plots : bool
        Plot diagnostic figure around the stitch.

    Returns
    -------
    strains : dict
        ``{ifo_name: stitched_strain}`` – numpy arrays.
    time : numpy.ndarray
        Common time axis.
    hp, hc : numpy.ndarray
        Stitched polarisations (+,×) in the source frame.
    """
    # ------------------------------------------------------------------
    # Phase & amplitude alignment derived from h+
    # ------------------------------------------------------------------
    n = min(len(hp1), len(hp2))
    hp1 = hp1[:n]
    hp2 = hp2[:n]
    t = hp1.sample_times[:n] if hasattr(hp1, "sample_times") else np.arange(n) * delta_t

    analytic1 = hilbert(hp1)
    analytic2 = hilbert(hp2)
    amp1 = np.abs(analytic1)
    amp2 = np.abs(analytic2)
    phase1 = np.unwrap(np.angle(analytic1))
    phase2 = np.unwrap(np.angle(analytic2))

    idx = np.argmin(np.abs(t - stitch_time))
    phase_shift = phase2[idx] - phase1[idx]
    amp_scale = amp1[idx] / amp2[idx]

    # Stitch + and × using *same* shift & scale
    hp, _ = _stitch_real(
        hp1,
        hp2,
        stitch_time,
        transition_length,
        delta_t,
        phase_shift,
        amp_scale,
        transition_type,
    )

    hc, _ = _stitch_real(
        hc1,
        hc2,
        stitch_time,
        transition_length,
        delta_t,
        phase_shift,
        amp_scale,
        transition_type,
    )

    # ------------------------------------------------------------------
    # Project onto detectors
    # ------------------------------------------------------------------
    strains = {}
    for ifo, (Fp, Fx) in detector_dict.items():
        # Allow scalar or array responses
        strains[ifo] = Fp * hp + Fx * hc

    # ------------------------------------------------------------------
    # Optional diagnostic plot
    # ------------------------------------------------------------------
    if show_plots:
        win = int(0.05 / delta_t)  # ±50 ms window
        i0 = np.argmin(np.abs(t - (stitch_time - 0.05)))
        i1 = i0 + 2 * win
        plt.figure(figsize=(10, 6))
        plt.plot(t[i0:i1], hp[i0:i1], label="h+ (stitched)")
        plt.plot(t[i0:i1], hc[i0:i1], label="h× (stitched)")
        plt.axvline(stitch_time, color="k", ls="--", label="stitch_time")
        plt.xlabel("Time [s]")
        plt.ylabel("Strain (source frame)")
        plt.title("Stitched polarisations around transition")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return strains, t, hp, hc
