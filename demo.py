#!/usr/bin/env python3

import os
import numpy as np
import bilby
from troye_model import troye_model, SEGMENT_START
import lalsimulation as lalsim
from bilby.gw.conversion import generate_mass_parameters
import matplotlib.pyplot as plt


# ────────── helpers ────────────────────────────────────────────────────
def mchirp_q_to_mass1_mass2(mchirp, q):
    if not 0 < q <= 1:
        raise ValueError("q must be in (0,1].")
    m1 = mchirp * (1 + q) ** (1 / 5) / q ** (3 / 5)
    return m1, q * m1

def network_snr(distance, inj, wfg, ifos, verbose=False):
    inj = inj.copy()
    inj["luminosity_distance"] = distance

    # ── NEW: catch waveform failures ────────────────────────
    try:
        pols = wfg.frequency_domain_strain(inj)
    except ValueError as e:                # guard from troye_model
        if verbose:
            print("SNR calc rejected:", e)
        return np.nan                      # let caller know it failed
    # ────────────────────────────────────────────────────────

    def snr2_manual(ifo):
        hfd = ifo.get_detector_response(pols, inj)
        df  = ifo.frequency_array[1] - ifo.frequency_array[0]
        return 4 * np.sum(np.abs(hfd)**2 / ifo.power_spectral_density_array) * df

    total = 0.0
    for ifo in ifos:
        try:
            snr2 = ifo.optimal_snr_squared(pols, inj)  # newer Bilby
        except AttributeError:
            snr2 = snr2_manual(ifo)                    # fallback
        if verbose:
            print(f"→ {ifo.name} SNR² = {snr2:.2f}")
        total += snr2
    return np.sqrt(total)

def convert(p):
    # Bilby calls this once with {} when computing the noise evidence
    if not p:
        return p.copy(), None          # ← still give it a new dict

    p = p.copy()                       # ← NEW: protect the caller’s dict

    # ── bookkeeping ─────────────────────────────────────────
    if {"mass_1", "mass_2"} <= p.keys():
        m1, m2 = p["mass_1"], p["mass_2"]
        p["chirp_mass"] = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        p["mass_ratio"] = m2 / m1          # keep q ≤ 1
    elif {"chirp_mass", "mass_ratio"} <= p.keys():
        m1, m2 = mchirp_q_to_mass1_mass2(p["chirp_mass"], p["mass_ratio"])
        p["mass_1"], p["mass_2"] = m1, m2
    # ────────────────────────────────────────────────────────
    if not (0 < p["mass_ratio"] <= 1):
        raise ValueError("q out of range")
    if not (5 <= p["lambda_2_pre"]  <= 4900):
        raise ValueError("λ₂_pre out of range")
    if not (5 <= p["lambda_2_post"] <= 4900):
        raise ValueError("λ₂_post out of range")
    
    return p, None

# ────────── global parameters ─────────────────────────────────────────-
DURATION, FS = 128, 2**12        # seconds, Hz
DT = 1 / FS
F_MIN, F_MAX = 20, 2**12        # Hz band
OUTDIR, LABEL = "Example_TroyeEvent_1", "Example_TroyeEvent_1"
os.makedirs(OUTDIR, exist_ok=True)

if __name__ == "__main__":
    
    # ────────── injection dictionary with randomness ───────────────────────
    seed = np.random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    print(f"Random seed for reproducibility: {seed}")
    
    # Generate random values with specified distributions and precision
    mass_ratio = np.round(np.random.uniform(1/7.5, 1/3.5), 3)
    chirp_mass = np.round(np.random.uniform(2.000, 3.400), 3)
    lambda_2_pre = np.round(np.random.uniform(10.0, 4900.0), 1)
    lambda_2_post = np.round(np.random.uniform(10.0, 4900.0), 1)
    chi_1 = np.round(np.random.uniform(0.00, 0.99), 2)
    ra = np.round(np.random.uniform(0, 2 * np.pi), 2)
    
    # Cosine distribution for declination
    dec = np.round(np.arcsin(np.random.uniform(-1, 1)) if np.random.rand() < 0.5 
                   else -np.arcsin(np.random.uniform(-1, 1)), 2)
    
    # Sine distribution for theta_jn
    theta_jn = np.round(np.arccos(np.random.uniform(-1, 1)), 2)
    
    luminosity_distance = np.round(np.random.uniform(10.0, 99.9), 1)
    
    inj = {
        "mass_ratio": mass_ratio,
        "chirp_mass": chirp_mass,
        "lambda_1": 0.0,
        "lambda_2_pre": lambda_2_pre,
        "lambda_2_post": lambda_2_post,
        "stitch_time": -0.04,
        "chi_1": chi_1,
        "chi_2": 0.0,
        "ra": ra,
        "dec": dec,
        "theta_jn": theta_jn,
        "psi": 0.0,
        "phase": 0.0,
        "geocent_time": 1126259462.4,
        "luminosity_distance": luminosity_distance,
    }
    
    print(f"Random injection parameters generated:")
    print(f"  mass_ratio: {mass_ratio}")
    print(f"  chirp_mass: {chirp_mass}")
    print(f"  lambda_2_pre: {lambda_2_pre}")
    print(f"  lambda_2_post: {lambda_2_post}")
    print(f"  chi_1: {chi_1}")
    print(f"  ra: {ra}")
    print(f"  dec: {dec}")
    print(f"  theta_jn: {theta_jn}")
    print(f"  luminosity_distance: {luminosity_distance}")

    inj["mass_1"], inj["mass_2"] = mchirp_q_to_mass1_mass2(inj["chirp_mass"], inj["mass_ratio"])
    inj["total_mass"] = inj["mass_1"] + inj["mass_2"]
    inj = generate_mass_parameters(inj)

    # ────────── Waveform Generator ───────────────────────────
    wg = bilby.gw.WaveformGenerator(
        duration=DURATION,
        sampling_frequency=FS,
        start_time=SEGMENT_START,
        time_domain_source_model=troye_model,
        parameter_conversion=convert,
    )

    print("WG conversion =", wg.parameter_conversion)

    # ────────── Interferometers & Gaussian noise ───────────────────────────
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    frequencies = np.linspace(0, FS / 2, int(FS * DURATION / 2) + 1)
    psd_array   = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(f) for f in frequencies])
    psd_array[0] = psd_array[1]
    psd = bilby.gw.detector.PowerSpectralDensity(frequency_array=frequencies,
                                                psd_array=psd_array)

    for ifo in ifos:
        ifo.minimum_frequency, ifo.maximum_frequency = F_MIN, F_MAX

        try:
            ifo.set_strain_data_from_power_spectral_density(
                sampling_frequency=FS,
                duration=DURATION,
                start_time=SEGMENT_START,
                power_spectral_density=psd,
            )
        except TypeError:
            ifo.power_spectral_density = psd
            ifo.set_strain_data_from_power_spectral_density(
                sampling_frequency=FS,
                duration=DURATION,
                start_time=SEGMENT_START,
            )

    ifos.inject_signal(waveform_generator=wg, parameters=inj)

    # ────────── priors ─────────────────────────────────────────────────────
    priors = bilby.core.prior.PriorDict()
    priors["chirp_mass"] = bilby.core.prior.Uniform(2.0, 3.4, name="chirp_mass")
    priors["lambda_2_pre"] = bilby.core.prior.Uniform(10, 4900, name="lambda_2_pre")
    priors["lambda_2_post"] = bilby.core.prior.Uniform(10, 4900, name="lambda_2_post")

    for key in [
        "mass_ratio", "lambda_1", "chi_1", "chi_2", "ra", "dec", "psi",
        "theta_jn", "phase", "geocent_time", "luminosity_distance",
    ]:
        priors[key] = bilby.core.prior.DeltaFunction(inj[key])

    # ────────── likelihood & sampler ───────────────────────────────────────
    print("Injection complete. Launching PE with Dynesty...")
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=wg,
    )

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=100,
        npool=48,
        resume=True,
        checkpoint_interval=300,
        label=LABEL,
        outdir=OUTDIR,
        dlogz=0.5,
    )

    result.plot_corner()
    print(f"Results: {os.path.join(OUTDIR, LABEL + '_result.json')}")
