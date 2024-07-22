# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Generates synthetic OOD data for testing.
"""
import os

import numpy as np
from config import (BG_MIX_FILE, DATA_DIR, FG_SEED_FILE, MIX_TRAIN_FG_LAMBDA,
                    OOD_FG_MIX_SAMPLES, OOD_PROP_RANGE, OOD_PROP_SAMPLES,
                    OOD_SEED_FILE, OOD_SPS, RANDOM_SEED, STATIC_SYNTH_BG_CPS,
                    STATIC_SYNTH_LIVE_TIME_RANGE,
                    STATIC_SYNTH_LIVE_TIME_SAMPLING, STATIC_SYNTH_SNR_SAMPLING,
                    STATIC_SYNTH_TEST_SNR_RANGE, TARGET_BINS, TARGET_ECAL)
from riid.data.sampleset import SampleSet, read_hdf
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer

# Set rng
rng = np.random.default_rng(seed=RANDOM_SEED)

# Load in data
fg_seeds_ss = read_hdf(FG_SEED_FILE)
fg_seeds_ss, _ = fg_seeds_ss.split_fg_and_bg()
fg_seeds_ss.drop_sources_columns_with_all_zeros()

ood_seeds_ss = read_hdf(OOD_SEED_FILE)
ood_seeds_ss, _ = ood_seeds_ss.split_fg_and_bg()
ood_seeds_ss.drop_sources_columns_with_all_zeros()

bg_mixtures_ss = read_hdf(BG_MIX_FILE)

# Get expected source contributions
source_counts = {
    x.split(",")[0]: v
    for x, v in zip(
        fg_seeds_ss.sources.columns.get_level_values("Seed").values,
        fg_seeds_ss.info.total_counts
    )
}
Z = np.array(list(source_counts.values()))
expected_props = Z / Z.sum()

# Preprocessing
fg_seeds_ss = fg_seeds_ss.as_ecal(*TARGET_ECAL)
fg_seeds_ss.downsample_spectra(target_bins=TARGET_BINS)
fg_seeds_ss.normalize()

ood_seeds_ss = ood_seeds_ss.as_ecal(*TARGET_ECAL)
ood_seeds_ss.downsample_spectra(target_bins=TARGET_BINS)
ood_seeds_ss.normalize()

# Generate data
expected_anomaly_props = np.linspace(
    OOD_PROP_RANGE[0],
    OOD_PROP_RANGE[1],
    OOD_PROP_SAMPLES)
dirichlet_alphas = [
    np.append((1.0 - each)*expected_props, each)
    for each in expected_anomaly_props
]

for i in range(ood_seeds_ss.n_samples):
    ood_seed_ss = ood_seeds_ss[i]
    ood_seed_ss.drop_sources_columns_with_all_zeros()
    seeds_ss = SampleSet()
    seeds_ss.concat([fg_seeds_ss, ood_seed_ss])

    fg_mixtures_ss = SampleSet()
    for each in dirichlet_alphas:
        mix_ss = SeedMixer(
            seeds_ss=seeds_ss,
            mixture_size=seeds_ss.n_samples,
            dirichlet_alpha=each*MIX_TRAIN_FG_LAMBDA,
            random_state=RANDOM_SEED
        ).generate(OOD_FG_MIX_SAMPLES)
        fg_mixtures_ss.concat(mix_ss)

    static_syn = StaticSynthesizer(
        samples_per_seed=OOD_SPS,
        bg_cps=STATIC_SYNTH_BG_CPS,
        live_time_function=STATIC_SYNTH_LIVE_TIME_SAMPLING,
        live_time_function_args=STATIC_SYNTH_LIVE_TIME_RANGE,
        snr_function=STATIC_SYNTH_SNR_SAMPLING,
        snr_function_args=STATIC_SYNTH_TEST_SNR_RANGE,
        rng=rng
    )

    ood_fg_ss, _ = static_syn.generate(
        fg_mixtures_ss,
        bg_mixtures_ss
    )

    ood_fg_ss.drop_spectra_with_no_contributors()
    ood_fg_ss.clip_negatives()

    ood_isotope = ood_seed_ss.get_labels()[0]
    output_path = os.path.join(DATA_DIR, f"{ood_isotope}_ood_test.h5")
    ood_fg_ss.to_hdf(output_path)
