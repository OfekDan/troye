# TROYE: NSBH waveforms with a late-inspiral phase transition (Bilby)

This repository contains a lightweight **phenomenological** waveform model, **TROYE** (Transitional Representation Of varYing Equation-of-state), built to simulate a **rapid change in the neutron-star tidal deformability** during a neutron-star–black-hole (NSBH) inspiral.

At its core, TROYE:

1. Generates two baseline time-domain NSBH waveforms using **`IMRPhenomNSBH`**:
   - a *pre-transition* waveform with tidal deformability **`Λ_pre`**
   - a *post-transition* waveform with tidal deformability **`Λ_post`**
2. **Phase- and amplitude-aligns** the post-transition waveform at a chosen transition time.
3. **Blends** the two signals over a finite time window to avoid discontinuities.
4. Exposes a **Bilby-compatible** `time_domain_source_model` that Bilby can FFT and use in standard GW likelihoods.

Key files:

- `troye_model.py` — Bilby source model `troye_model()` (time domain)
- `stitch_waveforms.py` — stitching/alignment helper `stitch_event()`
- `demo.py` — end-to-end injection + parameter estimation example

---

## Requirements

You need a Python environment that provides:

- `bilby` (GW inference)
- `lalsuite` / `lalsimulation` (PSD helper in `demo.py`)
- `pycbc` (waveform generation via `get_td_waveform`)
- `numpy`, `scipy`, `matplotlib`

> TROYE relies on `pycbc.waveform.get_td_waveform(approximant="IMRPhenomNSBH")`. Make sure your PyCBC/LAL install supports this approximant.

---

## Quickstart

### 1) Generate a TROYE waveform in Bilby

```python
import bilby
import troye_model as troye

# Choose a segment where merger (defined below) occurs at t=0.
# If you change duration or sampling frequency, update start_time accordingly.
DURATION = 96.0
FS = 2**12
DT = 1 / FS
START_TIME = -DURATION + DT

# (Optional) configure the transition hyper-parameters (see below)
troye.STITCH_T = -0.04           # seconds before merger (t_stitch)
troye.TRANS_LEN = 1e-2           # seconds (tau)
troye.TRANS_SHAPE = "raised_cosine"  # window shape w(t)

wg = bilby.gw.WaveformGenerator(
    duration=DURATION,
    sampling_frequency=FS,
    start_time=START_TIME,
    time_domain_source_model=troye.troye_model,
    # optional but recommended: ensure consistent mass bookkeeping
    # (see convert() in demo.py for one working pattern)
    parameter_conversion=None,
)

params = dict(
    chirp_mass=2.7,               # Msun
    mass_ratio=0.2,               # q = m_NS / m_BH (<= 1)
    lambda_2_pre=400.0,           # Λ_pre
    lambda_2_post=2000.0,         # Λ_post
    luminosity_distance=100.0,    # Mpc
    # optional tapering before FFT (reduces spectral leakage)
    taper="startend",
    tukey_roll_off=0.1,
)

pols_fd = wg.frequency_domain_strain(params)  # {"plus": ..., "cross": ...}
```

### 2) Run the full injection + PE example

```bash
python demo.py
```

`demo.py`:

- draws a random NSBH injection,
- builds a Bilby `WaveformGenerator` using `troye_model`,
- injects into simulated H1/L1 Gaussian noise,
- runs `dynesty` nested sampling,
- outputs a corner plot and a `*_result.json`.

---

## Model definition (paper-consistent)

TROYE assumes the strain can be represented by two baseline tidal responses,

- `h_pre(t)` with `Λ_pre`
- `h_post(t)` with `Λ_post`

with an effective change at time **`t_stitch`**. The final stitched waveform is

- `h(t) = h_pre(t)` for `t < t_-`
- `h(t) = (1-w(t)) h_pre(t) + w(t) h_post_aligned(t)` for `t_- <= t <= t_+`
- `h(t) = h_post_aligned(t)` for `t > t_+`

where `t_± = t_stitch ± τ/2`, `τ` is the blending-window duration, and `w(t)` is a smooth transition function.

### Merger / time convention

- **Merger is defined as the peak amplitude of the *pre-transition* waveform.**
- TROYE uses a **merger-referenced time grid** where merger occurs at **`t = 0`**.
- The default transition time `t_stitch` is negative (e.g. `-0.04 s` means 40 ms before merger).

---

## Parameters

### TROYE-specific parameters

| Parameter | Meaning | Where it lives in this repo |
|---|---|---|
| `lambda_2_pre` | NS tidal deformability **before** the transition (`Λ_pre`) | sampled parameter to `troye_model()` |
| `lambda_2_post` | NS tidal deformability **after** the transition (`Λ_post`) | sampled parameter to `troye_model()` |
| `t_stitch` | transition epoch relative to merger (`t=0`), typically `< 0` | `troye_model.STITCH_T` (seconds) |
| `tau` | blending-window duration | `troye_model.TRANS_LEN` (seconds) |
| `w` | transition/window function shape | `troye_model.TRANS_SHAPE` (string) |

**Window choices** (`w` / `transition_type`) are implemented in `stitch_waveforms._blend_window()`:

- `raised_cosine` (default)
- `tukey`
- `linear`
- `hanning`
- `hamming`

> **Current implementation note:** `t_stitch`, `tau`, and `w` are set as **module-level constants** in `troye_model.py` and are not yet passed as per-event sampled parameters. You can still:
>
> - set them **globally for a run** by assigning `troye_model.STITCH_T`, `troye_model.TRANS_LEN`, `troye_model.TRANS_SHAPE` before building the `WaveformGenerator`.
> - promote them to **sampled parameters** by reading them from `kwargs` (or adding them to the function signature) and forwarding them to `stitch_event()`.

### Usual intrinsic parameters (NSBH)

TROYE’s baseline waveforms are generated with `IMRPhenomNSBH`, so the natural intrinsic parameters are:

- masses: (`chirp_mass`, `mass_ratio`) or (`mass_1`, `mass_2`)
  - in this repo the Bilby-facing interface is **`chirp_mass`** and **`mass_ratio`**
  - by convention `q = m_NS / m_BH <= 1`
- aligned spins: `chi_1` (BH), `chi_2` (NS)
- tidal deformabilities: `lambda_1 = 0` (BH), `lambda_2 = Λ_NS`

> **Current implementation note:** the shipped `troye_model()` currently forwards only `(m1, m2, Λ, d_L)` into the PyCBC call. If you want spins (and/or inclination/phase) to matter, extend `troye_model()` to pass them through to `_phenom_nsbh_td()`.

### Usual extrinsic parameters (Bilby)

For detector response and likelihood evaluation in Bilby you will typically provide:

- `luminosity_distance` (Mpc)
- `ra`, `dec` (sky location)
- `psi` (polarization angle)
- `theta_jn` (inclination)
- `phase` (coalescence phase)
- `geocent_time` (GPS time)

Bilby uses these to compute antenna patterns, time delays, and the detector strain via `ifo.get_detector_response(...)`.

---

## Null hypothesis (no transition)

To recover a standard constant-tidal NSBH waveform within the same interface:

- set `lambda_2_pre == lambda_2_post` (equivalently `ΔΛ = 0`).

This is useful for Bayes-factor comparisons between a transitional hypothesis and a static tidal hypothesis.

---

## Practical tips / gotchas

- **Sampling frequency must match TROYE’s internal `delta_t`.** In this repo, `troye_model.DT = 2**-12` seconds (FS = 4096 Hz). If you change `FS`, update `DT` (and ideally thread it through to `stitch_event(delta_t=...)`).
- **Segment timing:** for a clean merger-referenced segment, set `start_time = -duration + 1/FS` so the time array ends at ~0.
- **Domain validity:** `IMRPhenomNSBH` (and tidal modeling in general) has regions where waveform generation can fail. In `demo.py`, the `convert()` function shows one pattern for rejecting invalid proposals early.

---

## Citation

If you use or adapt this implementation in academic work, please cite the accompanying TROYE paper that describes the stitching construction and inference setup.
The full results catalog of the simulation campaign descirbed in the paper is given [here](paper_results/Catalog.pdf).
