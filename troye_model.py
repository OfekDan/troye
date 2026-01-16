from __future__ import annotations
import numpy as np
from pycbc.waveform import get_td_waveform
from scipy.signal.windows import tukey
from stitch_waveforms import stitch_event
#import matplotlib
#matplotlib.use("QtAgg")  # or "TkAgg", "MacOSX" (mac only)
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Constants (can be overridden via waveform_arguments)
# -----------------------------------------------------------------------------
DT            = 2 ** -12           # 
DURATION      = 96.0               # seconds – length of Bilby time array
SEGMENT_START = -DURATION + DT     # so that t=0 coincides with merger
F_LOW         = 20.0               # Hz – low‑freq cut‑off for IMRPhenomNSBH
STITCH_T      = -0.04              # seconds before merger where Λ changes (t=0 at merger)
TRANS_LEN     = 1e-2               # 10 ms raised‑cosine blend
TRANS_SHAPE   = "raised_cosine"    # blend window

# -----------------------------------------------------------------------------
# Internal helper – generate a single tidal Phenom NSBH waveform (plus, cross)
# -----------------------------------------------------------------------------

def _phenom_nsbh_td(m1, m2, lambda2_ns, distance, chi1z=0, chi2z=0):
    # ensure m1 is the BH, m2 the NS
    if m1 < m2:
        m1, m2 = m2, m1
    try:
        hp, hc = get_td_waveform(
            approximant="IMRPhenomNSBH",
            mass1=m1,  spin1z=chi1z,  lambda1=0,
            mass2=m2,  spin2z=chi2z,  lambda2=lambda2_ns,
            f_lower=F_LOW,   delta_t=DT,  distance=distance,
            inclination=0,
        )
    except Exception as e:
        import traceback, sys
        print("\n---- PyCBC/LAL FAILURE ----")
        traceback.print_exc(file=sys.stdout)
        print("---------------------------\n", flush=True)
        raise        # re-raise so your guard still fires
    return hp, hc


# -----------------------------------------------------------------------------
#  Main Bilby source‑model function (to be passed to WaveformGenerator)
# -----------------------------------------------------------------------------

def troye_model(
    t_arr: np.ndarray,
    chirp_mass: float,
    mass_ratio: float,
    lambda_2_pre: float,
    lambda_2_post: float,
    luminosity_distance: float,
    **kwargs,
):
    """
    TROYE: Toy Representation of varYing Equation-of-state.
    Stitched NSBH waveform on a merger-referenced time grid.

    Parameters
    ----------
    t_arr : np.ndarray
        Monotonic time samples that terminate at ``t≈0`` (merger). The first
        sample is typically ``SEGMENT_START``.
    """

    # --- recover component masses ------------------------------------------
    eta = mass_ratio / (1 + mass_ratio) ** 2  # symmetric mass ratio
    total_mass = chirp_mass / eta ** (3 / 5)
    m1 = total_mass / (1 + mass_ratio)
    m2 = total_mass - m1

    # --- generate pre‑ and post‑transition segments ------------------------
    hp_pre, hc_pre = _phenom_nsbh_td(m1, m2, lambda_2_pre, luminosity_distance)
    hp_post, hc_post = _phenom_nsbh_td(m1, m2, lambda_2_post, luminosity_distance)

    # --- stitch them -------------------------------------------------------
    _, t_det, hp_st, hc_st = stitch_event(
        hp_pre,
        hc_pre,
        hp_post,
        hc_post,
        detector_dict={"IFO": (1.0, 0.0)},
        stitch_time=STITCH_T,
        transition_length=TRANS_LEN,
        transition_type=TRANS_SHAPE,
        show_plots=False,
    )

    # Ensure Bilby time grid is referenced to merger (t=0)
    t_det_zero = t_det[np.argmin(np.abs(t_det))]
    if not np.isclose(t_det_zero, 0.0, atol=DT):
        raise ValueError(
            "Expect stitched waveform to include merger sample at t=0; "
            f"closest sample is at {t_det_zero:.6f} s."
        )
    '''if not np.isclose(t_arr[-1], 0.0, atol=DT):
        raise ValueError(
            "Bilby time array must be merger-referenced (ends at t≈0)."
        )'''

    # interpolate onto Bilby's grid
    h_plus = np.interp(t_arr, t_det, hp_st, left=0.0, right=0.0)
    h_cross = np.interp(t_arr, t_det, hc_st, left=0.0, right=0.0)

    # Optional tapering (to reduce spectral leakage when FFT'd by Bilby)
    taper_kind = kwargs.get("taper", None)
    if taper_kind in ("startend", True):
        alpha = float(kwargs.get("tukey_roll_off", 0.0))
        win = tukey(len(t_arr), alpha=alpha)
        h_plus *= win
        h_cross *= win

    #return {"plus": h_plus, "cross": h_cross}
    return {"plus": h_plus, "cross": h_cross, "time": t_arr}