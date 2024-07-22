# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Compute the inference time for the models on the test set.
"""
import time

import numpy as np
from config import STATIC_SYNTH_BG_CPS, TEST_FILE
from riid.data.sampleset import read_hdf
from utils import load_final_models

ind_synth_test_ss = read_hdf(TEST_FILE)

# Load in final model for each unsupervised loss (either locally or with W&B).
best_runs, best_models = load_final_models()

num_trials = 10

for idx, model in enumerate(best_models):
    times = []
    for trial in range(num_trials):
        start = time.time()
        best_models[idx].predict(ind_synth_test_ss, bg_cps=STATIC_SYNTH_BG_CPS)
        end = time.time()
        times.append((end - start) / ind_synth_test_ss.n_samples)

    print(
        f"Unsupervised Loss: {list(best_runs.keys())[idx]}, " +
        f"Mean: {np.mean(times)} s, STD: {np.std(times)} s")
