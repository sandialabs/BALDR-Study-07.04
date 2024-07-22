# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Generates synthetic in-distribution data for training, testing, and validation.
"""
import numpy as np
from config import (BG_MIX_FILE, BG_SEED_FILE, FG_SEED_FILE,
                    MIX_TEST_FG_SAMPLES, MIX_TRAIN_BG_ALPHA,
                    MIX_TRAIN_BG_SAMPLES, MIX_TRAIN_BG_SIZE,
                    MIX_TRAIN_FG_LAMBDA, MIX_TRAIN_FG_SAMPLES, RANDOM_SEED,
                    STATIC_SYNTH_BG_CPS, STATIC_SYNTH_LIVE_TIME_RANGE,
                    STATIC_SYNTH_LIVE_TIME_SAMPLING, STATIC_SYNTH_LONG_BG_CPS,
                    STATIC_SYNTH_SNR_RANGE, STATIC_SYNTH_SNR_SAMPLING,
                    STATIC_SYNTH_SPS, STATIC_SYNTH_TEST_SNR_RANGE,
                    STATIC_SYNTH_VAL_SNR_RANGE, TARGET_BINS, TARGET_ECAL,
                    TEST_FG_MIX_FILE, TEST_FILE, TRAIN_FG_MIX_FILE, TRAIN_FILE,
                    TUNING_VAL_FG_MIX_FILE, TUNING_VAL_FILE)
from riid.data.sampleset import read_hdf
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer

# Set rng
rng = np.random.default_rng(seed=RANDOM_SEED)

# Load in seeds
fg_seeds_ss = read_hdf(FG_SEED_FILE)
fg_seeds_ss, _ = fg_seeds_ss.split_fg_and_bg()
fg_seeds_ss.drop_sources_columns_with_all_zeros()

bg_seeds_ss = read_hdf(BG_SEED_FILE)
_, bg_seeds_ss = bg_seeds_ss.split_fg_and_bg()
bg_seeds_ss.drop_sources_columns_with_all_zeros()

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

bg_seeds_ss = bg_seeds_ss.as_ecal(*TARGET_ECAL)
bg_seeds_ss.downsample_spectra(target_bins=TARGET_BINS)
bg_seeds_ss.normalize()

static_syn = StaticSynthesizer(
    samples_per_seed=STATIC_SYNTH_SPS,
    bg_cps=STATIC_SYNTH_BG_CPS,
    live_time_function=STATIC_SYNTH_LIVE_TIME_SAMPLING,
    live_time_function_args=STATIC_SYNTH_LIVE_TIME_RANGE,
    snr_function=STATIC_SYNTH_SNR_SAMPLING,
    snr_function_args=STATIC_SYNTH_SNR_RANGE,
    long_bg_live_time=STATIC_SYNTH_LONG_BG_CPS,
    rng=rng
)

# Background
mixed_bg_seeds_ss = SeedMixer(
    bg_seeds_ss,
    mixture_size=MIX_TRAIN_BG_SIZE,
    dirichlet_alpha=MIX_TRAIN_BG_ALPHA,
    random_state=RANDOM_SEED
).generate(MIX_TRAIN_BG_SAMPLES)
mixed_bg_seeds_ss.to_hdf(BG_MIX_FILE)

# Train
train_mixed_fg_seeds_ss = SeedMixer(
    fg_seeds_ss,
    mixture_size=fg_seeds_ss.n_samples,
    dirichlet_alpha=expected_props * MIX_TRAIN_FG_LAMBDA,
    random_state=RANDOM_SEED
).generate(MIX_TRAIN_FG_SAMPLES)
train_mixed_fg_seeds_ss.to_hdf(TRAIN_FG_MIX_FILE)
train_ss, _ = static_syn.generate(
    fg_seeds_ss=train_mixed_fg_seeds_ss,
    bg_seeds_ss=mixed_bg_seeds_ss
)
train_ss.drop_spectra_with_no_contributors()
train_ss.clip_negatives()
train_ss.to_hdf(TRAIN_FILE)

# Validation
static_syn.snr_function_args = STATIC_SYNTH_VAL_SNR_RANGE
val_mixed_fg_seeds_ss = SeedMixer(
    fg_seeds_ss,
    mixture_size=fg_seeds_ss.n_samples,
    dirichlet_alpha=expected_props * MIX_TRAIN_FG_LAMBDA,
    random_state=RANDOM_SEED
).generate(MIX_TEST_FG_SAMPLES)
val_mixed_fg_seeds_ss.to_hdf(TUNING_VAL_FG_MIX_FILE)
val_ss, _ = static_syn.generate(
    fg_seeds_ss=val_mixed_fg_seeds_ss,
    bg_seeds_ss=mixed_bg_seeds_ss
)
val_ss.drop_spectra_with_no_contributors()
val_ss.clip_negatives()
val_ss.to_hdf(TUNING_VAL_FILE)

# Test
static_syn.snr_function_args = STATIC_SYNTH_TEST_SNR_RANGE
test_mixed_fg_seeds_ss = SeedMixer(
    fg_seeds_ss,
    mixture_size=fg_seeds_ss.n_samples,
    dirichlet_alpha=expected_props * MIX_TRAIN_FG_LAMBDA,
    random_state=RANDOM_SEED
).generate(MIX_TEST_FG_SAMPLES)
test_mixed_fg_seeds_ss.to_hdf(TEST_FG_MIX_FILE)
test_ss, _ = static_syn.generate(
    fg_seeds_ss=test_mixed_fg_seeds_ss,
    bg_seeds_ss=mixed_bg_seeds_ss
)
test_ss.drop_spectra_with_no_contributors()
test_ss.clip_negatives()
test_ss.to_hdf(TEST_FILE)
