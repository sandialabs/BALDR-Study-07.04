# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Contains global constants for experiment configuration.
"""
import os

# Directories
DATA_DIR = "./data/"  # DATA_DIR should already exist with seed files
PNNL_DATA_DIR = os.path.join(DATA_DIR, "pnnl")
os.makedirs(PNNL_DATA_DIR, exist_ok=True)
IMAGE_DIR = "./imgs/"
os.makedirs(IMAGE_DIR, exist_ok=True)
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)
RUN_DIR = "./runs"
os.makedirs(RUN_DIR, exist_ok=True)

# Seed files
FG_SEED_FILE = os.path.join(DATA_DIR, "U-235_th_FALCON_11MIN_fg_seeds.h5")
BG_SEED_FILE = os.path.join(DATA_DIR, "U-235_th_FALCON_11MIN_bg_seeds.h5")
OOD_SEED_FILE = os.path.join(DATA_DIR, "U-235_th_FALCON_11MIN_ood_seeds.h5")

# Synthetic training/testing files
BG_MIX_FILE = os.path.join(DATA_DIR, "bg_mixtures.h5")
TRAIN_FILE = os.path.join(DATA_DIR, "train.h5")
TRAIN_FG_MIX_FILE = os.path.join(DATA_DIR, "train_fg_mixtures.h5")
TEST_FILE = os.path.join(DATA_DIR, "test.h5")
TEST_FG_MIX_FILE = os.path.join(DATA_DIR, "test_fg_mixtures.h5")
TUNING_VAL_FILE = os.path.join(DATA_DIR, "tuning_val.h5")
TUNING_VAL_FG_MIX_FILE = os.path.join(DATA_DIR, "tuning_val_fg_mixtures.h5")

# Measured data from PNNL
PNNL_MEASUREMENTS_FILE = os.path.join(DATA_DIR, "pnnl_measurements.h5")
PNNL_FG_FILE = os.path.join(DATA_DIR, "pnnl_fg.h5")
PNNL_BG_FILE = os.path.join(DATA_DIR, "pnnl_bg.h5")

# Random seed for reproducibility
RANDOM_SEED = 42

# Seed preprocessing config
TARGET_BINS = 512
TARGET_ECAL = (0.0, 1500.0, 0.0, 0.0, 0.0)

# SeedMixer config
MIX_TRAIN_BG_SIZE = 4
MIX_TRAIN_BG_ALPHA = 2
MIX_TRAIN_BG_SAMPLES = 1
MIX_TRAIN_FG_SAMPLES = 200000
MIX_TRAIN_FG_LAMBDA = 500
MIX_TEST_FG_SAMPLES = 5000

# StaticSynthesizer config
STATIC_SYNTH_SPS = 1
STATIC_SYNTH_BG_CPS = 50
STATIC_SYNTH_LONG_BG_CPS = 600
STATIC_SYNTH_LIVE_TIME_RANGE = (600, 600)
STATIC_SYNTH_LIVE_TIME_SAMPLING = "uniform"
STATIC_SYNTH_SNR_SAMPLING = "log10"
STATIC_SYNTH_SNR_RANGE = (50, 2000)
STATIC_SYNTH_VAL_SNR_RANGE = (50, 2000)
STATIC_SYNTH_TEST_SNR_RANGE = (1, 2000)

# OOD test config
OOD_PROP_RANGE = (0.01, 0.99)
OOD_PROP_SAMPLES = 20
OOD_FG_MIX_SAMPLES = 50
OOD_SPS = 50

# Default training config
MODEL_TRAIN_ON_STATIC_SYNTH_DATA = True
MODEL_TRAIN_ON_GPU = False
MODEL_USE_WANDB = False  # Set to True if using W&B and modify next two variables
MODEL_WANDB_HOST = None  # Set URL to W&B
MODEL_WANDB_PROJECT = None  # Set project name in W&B
MODEL_OPTIMIZER = "Adam"
MODEL_OPTIMIZER_KWARGS = {"epsilon": 1.6389692779515138e-06}
MODEL_INIT_LR = 0.0002866875212513434
MODEL_BATCH_SIZE = 500
MODEL_EPOCHS = 300
MODEL_BETA = 0.38560892324748186
MODEL_UNSUP_LOSS = "jsd"
MODEL_SUP_LOSS = "sparsemax"
MODEL_NORMALIZE_SUP_LOSS = False
MODEL_SUP_NORMALIZE_SCALER = 1
MODEL_HIDDEN_LAYERS = (808, 407,)
MODEL_HIDDEN_LAYER_ACTIVATION = "relu"
MODEL_KERNEL_L1_REG = 0.0
MODEL_KERNEL_L2_REG = 0.0
MODEL_ACTIVITY_L1_REG = 0.0
MODEL_ACTIVITY_L2_REG = 0.0
MODEL_DROPOUT = 0.11289087980288798
MODEL_TARGET_LEVEL = "Isotope"
MODEL_PATIENCE = 15
MODEL_MIN_DELTA = 1e-7
MODEL_LR_SCHED_PATIENCE = 10
MODEL_LR_SCHED_MIN_DELTA = 1e-7
MODEL_VAL_SPLIT = 0.2
MODEL_OOD_FPR = 0.05
MODEL_SPLINE_BINS = 60
MODEL_SPLINE_K = 3
MODEL_SPLINE_S = 0

# Adjusted hyperparamater search values
HP_SWEEP_PATIENCE = 5
HP_SWEEP_PATIENCE_MIN_DELTA = 1e-7
HP_SWEEP_LR_SCHED_PATIENCE = 4
HP_SWEEP_LR_SCHED_PATIENCE_MIN_DELTA = 1e-7
HP_SWEEP_EPOCHS = 50
HP_SWEEP_TRIALS = 250

# Path to final models for evaluation
EVAL_USE_LOCAL = True
EVAL_LOCAL_BEST_MODELS = {  # dictionary with model names
    "chi_squared": "lpe_sparsemax_chi_squared_1.6576298943164755e-09_20240425-084353.onnx",
    "jsd": "lpe_sparsemax_jsd_0.38560892324748186_20231106-072915.onnx",
    "pnll": "lpe_sparsemax_poisson_nll_3.452109518587601e-07_20231106-073216.onnx",
    "sse": "lpe_sparsemax_sse_2.4952671831088513e-10_20231106-073501.onnx"
}
EVAL_WANDB_BEST_MODELS = {  # fill in with run path from W&B
    "chi_squared": None,
    "jsd": None,
    "pnll": None,
    "sse": None
}
