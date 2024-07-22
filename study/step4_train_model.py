# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Trains an LPE model.
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from config import (DATA_DIR, FG_SEED_FILE, MIX_TEST_FG_SAMPLES,
                    MIX_TRAIN_BG_ALPHA, MIX_TRAIN_BG_SAMPLES,
                    MIX_TRAIN_BG_SIZE, MIX_TRAIN_FG_LAMBDA,
                    MIX_TRAIN_FG_SAMPLES, MODEL_ACTIVITY_L1_REG,
                    MODEL_ACTIVITY_L2_REG, MODEL_BATCH_SIZE, MODEL_BETA,
                    MODEL_DIR, MODEL_DROPOUT, MODEL_EPOCHS,
                    MODEL_HIDDEN_LAYER_ACTIVATION, MODEL_HIDDEN_LAYERS,
                    MODEL_INIT_LR, MODEL_KERNEL_L1_REG, MODEL_KERNEL_L2_REG,
                    MODEL_LR_SCHED_MIN_DELTA, MODEL_LR_SCHED_PATIENCE,
                    MODEL_MIN_DELTA, MODEL_NORMALIZE_SUP_LOSS, MODEL_OOD_FPR,
                    MODEL_OPTIMIZER, MODEL_OPTIMIZER_KWARGS, MODEL_PATIENCE,
                    MODEL_SPLINE_BINS, MODEL_SPLINE_K, MODEL_SPLINE_S,
                    MODEL_SUP_LOSS, MODEL_SUP_NORMALIZE_SCALER,
                    MODEL_TARGET_LEVEL, MODEL_TRAIN_ON_GPU,
                    MODEL_TRAIN_ON_STATIC_SYNTH_DATA, MODEL_UNSUP_LOSS,
                    MODEL_USE_WANDB, MODEL_VAL_SPLIT, MODEL_WANDB_HOST,
                    MODEL_WANDB_PROJECT, OOD_FG_MIX_SAMPLES, OOD_PROP_RANGE,
                    OOD_PROP_SAMPLES, OOD_SEED_FILE, OOD_SPS,
                    STATIC_SYNTH_BG_CPS, STATIC_SYNTH_LIVE_TIME_RANGE,
                    STATIC_SYNTH_LIVE_TIME_SAMPLING, STATIC_SYNTH_LONG_BG_CPS,
                    STATIC_SYNTH_SNR_RANGE, STATIC_SYNTH_SNR_SAMPLING,
                    STATIC_SYNTH_SPS, STATIC_SYNTH_TEST_SNR_RANGE, TARGET_BINS,
                    TARGET_ECAL, TEST_FG_MIX_FILE, TEST_FILE,
                    TRAIN_FG_MIX_FILE, TRAIN_FILE)
from riid.data.sampleset import read_hdf
from riid.models.neural_nets import LabelProportionEstimator
from sklearn.metrics import mean_absolute_error as mae
from utils import (SaveTruthsandPredictionsCallback, animate_calibration,
                   generate_calibration_plot, generate_mae_vs_snr_plot,
                   generate_ood_heatmap,
                   generate_recon_error_vs_ood_contribution_plot,
                   generate_threshold_plot)
from wandb.keras import WandbMetricsLogger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)

if MODEL_TRAIN_ON_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # optionally select valid gpus
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

time_str = time.strftime("%Y%m%d-%H%M%S")
model_name = f"lpe_{MODEL_SUP_LOSS}_{MODEL_UNSUP_LOSS}_{MODEL_BETA}_{time_str}"

fg_seeds_ss = read_hdf(FG_SEED_FILE)
fg_seeds_ss, _ = fg_seeds_ss.split_fg_and_bg()
fg_seeds_ss.drop_sources_columns_with_all_zeros()
ood_seeds_ss = read_hdf(OOD_SEED_FILE)
ood_seeds_ss, _ = ood_seeds_ss.split_fg_and_bg()
ood_seeds_ss.drop_sources_columns_with_all_zeros()

fg_seeds_ss = fg_seeds_ss.as_ecal(*TARGET_ECAL)
fg_seeds_ss.downsample_spectra(target_bins=TARGET_BINS)
fg_seeds_ss.normalize()
ood_seeds_ss = ood_seeds_ss.as_ecal(*TARGET_ECAL)
ood_seeds_ss.downsample_spectra(target_bins=TARGET_BINS)
ood_seeds_ss.normalize()

if MODEL_USE_WANDB:
    wandb.login(host=MODEL_WANDB_HOST)
    wandb.init(
        project=MODEL_WANDB_PROJECT,
        config={
            "preprocessing": {
                "spectra_bins": TARGET_BINS,
                "spectra_ecal": TARGET_ECAL,
            },
            "SeedMixer": {
                "train_bg_size": MIX_TRAIN_BG_SIZE,
                "train_bg_alpha": MIX_TRAIN_BG_ALPHA,
                "train_bg_samples": MIX_TRAIN_BG_SAMPLES,
                "train_fg_samples": MIX_TRAIN_FG_SAMPLES,
                "train_fg_lambda": MIX_TRAIN_FG_LAMBDA,
                "test_fg_samples": MIX_TEST_FG_SAMPLES,
            },
            "StaticSynth": {
                "sps": STATIC_SYNTH_SPS,
                "bg_cps": STATIC_SYNTH_BG_CPS,
                "long_bg_cps": STATIC_SYNTH_LONG_BG_CPS,
                "live_time_range": STATIC_SYNTH_LIVE_TIME_RANGE,
                "live_time_sampling": STATIC_SYNTH_LIVE_TIME_SAMPLING,
                "snr_range": STATIC_SYNTH_SNR_RANGE,
                "snr_sampling": STATIC_SYNTH_SNR_SAMPLING,
                "snr_test_range": STATIC_SYNTH_TEST_SNR_RANGE
            },
            "OOD": {
                "OOD_prop_range": OOD_PROP_RANGE,
                "OOD_prop_samples": OOD_PROP_SAMPLES,
                "OOD_fg_mix_smples": OOD_FG_MIX_SAMPLES,
                "OOD_sps": OOD_SPS
            },
            "model": {
                "train_on_static_synth_data": MODEL_TRAIN_ON_STATIC_SYNTH_DATA,
                "train_on_GPU": MODEL_TRAIN_ON_GPU,
                "target_level": MODEL_TARGET_LEVEL,
                "beta": MODEL_BETA,
                "unsup_loss": MODEL_UNSUP_LOSS,
                "sup_loss": MODEL_SUP_LOSS,
                "normalize_sup_loss": MODEL_NORMALIZE_SUP_LOSS,
                "normalize_sup_loss_scaler": MODEL_SUP_NORMALIZE_SCALER,
                "optimizer": MODEL_OPTIMIZER,
                "optimizer_kwargs": MODEL_OPTIMIZER_KWARGS,
                "init_lr": MODEL_INIT_LR,
                "batch_size": MODEL_BATCH_SIZE,
                "epochs": MODEL_EPOCHS,
                "hidden_layers": MODEL_HIDDEN_LAYERS,
                "hidden_layer_activation": MODEL_HIDDEN_LAYER_ACTIVATION,
                "kernel_l1_reg": MODEL_KERNEL_L1_REG,
                "kernel_l2_reg": MODEL_KERNEL_L2_REG,
                "activity_l1_reg": MODEL_ACTIVITY_L1_REG,
                "activity_l2_reg": MODEL_ACTIVITY_L2_REG,
                "dropout": MODEL_DROPOUT,
                "patience": MODEL_PATIENCE,
                "patience_delta": MODEL_MIN_DELTA,
                "lr_patience": MODEL_LR_SCHED_PATIENCE,
                "lr_patience_delta": MODEL_LR_SCHED_MIN_DELTA,
                "val_split": MODEL_VAL_SPLIT,
                "ood_fpr": MODEL_OOD_FPR,
                "spline_bins": MODEL_SPLINE_BINS,
                "spline_k": MODEL_SPLINE_K,
                "spline_s": MODEL_SPLINE_S
            }
        }
    )
    run_dir = wandb.run.dir
    callback_dir = os.path.join(run_dir, "callbacks")
    if not os.path.exists(callback_dir):
        os.makedirs(callback_dir)

else:
    run_dir = MODEL_DIR
    callback_dir = os.path.join(run_dir, "callbacks")

if not os.path.exists(callback_dir):
    os.makedirs(callback_dir)

if MODEL_TRAIN_ON_STATIC_SYNTH_DATA:
    train_ss = read_hdf(TRAIN_FILE)
    test_ss = read_hdf(TEST_FILE)
else:
    train_ss = read_hdf(TRAIN_FG_MIX_FILE)
    test_ss = read_hdf(TEST_FG_MIX_FILE)

model = LabelProportionEstimator(
    hidden_layers=MODEL_HIDDEN_LAYERS,
    sup_loss=MODEL_SUP_LOSS,
    unsup_loss=MODEL_UNSUP_LOSS,
    beta=MODEL_BETA,
    fg_dict=None,
    optimizer=MODEL_OPTIMIZER,
    optimizer_kwargs=MODEL_OPTIMIZER_KWARGS,
    learning_rate=MODEL_INIT_LR,
    metrics=["mae"],
    hidden_layer_activation=MODEL_HIDDEN_LAYER_ACTIVATION,
    kernel_l1_regularization=MODEL_KERNEL_L1_REG,
    kernel_l2_regularization=MODEL_KERNEL_L2_REG,
    activity_l1_regularization=MODEL_ACTIVITY_L1_REG,
    activity_l2_regularization=MODEL_ACTIVITY_L2_REG,
    dropout=MODEL_DROPOUT,
    target_level=MODEL_TARGET_LEVEL,
    bg_cps=STATIC_SYNTH_BG_CPS,
    fit_spline=MODEL_TRAIN_ON_STATIC_SYNTH_DATA,
    ood_fp_rate=MODEL_OOD_FPR,
    spline_bins=MODEL_SPLINE_BINS,
    spline_k=MODEL_SPLINE_K,
    spline_s=MODEL_SPLINE_S
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=MODEL_LR_SCHED_PATIENCE,
        min_delta=MODEL_LR_SCHED_MIN_DELTA
    ),
    SaveTruthsandPredictionsCallback(
        test_ss.spectra,
        test_ss.sources,
        model_name,
        model.activation,
        callback_dir=callback_dir
    )
]

if MODEL_USE_WANDB:
    callbacks = callbacks + [
        WandbMetricsLogger(log_freq="epoch"),
    ]

history = model.fit(
    seeds_ss=fg_seeds_ss,
    ss=train_ss,
    batch_size=MODEL_BATCH_SIZE,
    epochs=MODEL_EPOCHS,
    validation_split=MODEL_VAL_SPLIT,
    callbacks=callbacks,
    patience=MODEL_PATIENCE,
    verbose=True,
    normalize_scaler=MODEL_SUP_NORMALIZE_SCALER,
    normalize_sup_loss=MODEL_NORMALIZE_SUP_LOSS,
    bg_cps=STATIC_SYNTH_BG_CPS,
    es_min_delta=MODEL_MIN_DELTA
)

model.save(os.path.join(run_dir, f"{model_name}.onnx"))

if MODEL_USE_WANDB:
    print("Running forward pass on train/test data...")
    model.predict(train_ss, bg_cps=STATIC_SYNTH_BG_CPS)
    train_true_props = train_ss.sources.values.flatten()
    train_pred_props = train_ss.prediction_probas.values.flatten()
    train_mae = mae(train_true_props, train_pred_props)

    model.predict(test_ss, bg_cps=STATIC_SYNTH_BG_CPS)
    test_true_props = test_ss.sources.values.flatten()
    test_pred_props = test_ss.prediction_probas.values.flatten()
    test_mae = mae(test_true_props, test_pred_props)

    wandb.log({
        "final_train_mae": train_mae,
        "final_test_mae": test_mae
    })

    print("Generating calibration plots for train/test data..")
    plt_max = np.max(np.concatenate((
        train_true_props,
        train_pred_props,
        test_true_props,
        test_pred_props
    ))) + 0.05
    plt_alpha = 0.5
    plt_figsize = (6.4, 4.8)
    plt_dpi = 200

    fig, ax = generate_calibration_plot(
        train_true_props,
        train_pred_props,
        plt_figsize,
        plt_dpi,
        plt_max,
        plt_alpha)

    train_calibration_img = wandb.Image(
        fig,
        caption="calibration plot for training dataset"
    )
    wandb.log({"train_calibration_plot": train_calibration_img})
    plt.close()

    fig, ax = generate_calibration_plot(
        test_true_props,
        test_pred_props,
        plt_figsize,
        plt_dpi,
        plt_max,
        plt_alpha)

    test_calibration_img = wandb.Image(
        fig,
        caption="calibration plot for testing dataset"
    )
    wandb.log({"test_calibration_plot": test_calibration_img})
    plt.close()

    print("Generating calibration animation for test data...")
    animate_calibration(
        callback_dir,
        run_dir,
        plt_max=plt_max
    )
    wandb.log({
        "calibration_animation": wandb.Video(
            os.path.join(run_dir, "calibration.gif"), fps=10
        )
    })

    if MODEL_TRAIN_ON_STATIC_SYNTH_DATA:
        print("Generating stats for remaining plots...")
        train_maes = np.array([
            mae(
                train_ss.sources.values[i, :],
                train_ss.prediction_probas.values[i, :]
            ) for i in range(train_ss.n_samples)
        ])
        train_lts = train_ss.info.live_time.values
        train_cnts = train_ss.info.total_counts.values
        train_snrs = np.array(train_cnts / np.sqrt(train_lts * STATIC_SYNTH_BG_CPS))

        test_maes = np.array([
            mae(
                test_ss.sources.values[i, :],
                test_ss.prediction_probas.values[i, :]
            ) for i in range(test_ss.n_samples)
        ])
        test_lts = test_ss.info.live_time.values
        test_cnts = test_ss.info.total_counts.values
        test_snrs = np.array(test_cnts / np.sqrt(test_lts * STATIC_SYNTH_BG_CPS))

        print("Generating mae vs. snr plots for train/test data...")
        fig, ax = generate_mae_vs_snr_plot(
            train_maes,
            train_snrs,
            plt_figsize,
            plt_dpi
        )
        train_mae_vs_snr_img = wandb.Image(
            fig,
            caption="mae vs. snr for training dataset"
        )
        wandb.log({"train_mae_vs_snr_plot": train_mae_vs_snr_img})
        plt.close()

        fig, ax = generate_mae_vs_snr_plot(
            test_maes,
            test_snrs,
            plt_figsize,
            plt_dpi
        )
        test_mae_vs_snr_img = wandb.Image(
            fig,
            caption="mae vs. snr for testing dataset"
        )
        wandb.log({"test_mae_vs_snr_plot": test_mae_vs_snr_img})
        plt.close()

        print("Generating OOD threshold function plot...")
        fig, ax = generate_threshold_plot(
            train_ss.info[f"unsup_{model.unsup_loss}_loss"].values,
            train_snrs,
            model.ood_threshold_func,
            plt_figsize,
            plt_dpi,
            plt_alpha
        )
        threshold_func_img = wandb.Image(
            fig,
            caption="threshold function determined from training dataset"
        )
        wandb.log({"threshold_func_plot": threshold_func_img})
        plt.close()

        ood_sources = list(ood_seeds_ss.get_labels())
        for ood_source in ood_sources:
            print(f"Generating OOD plot for {ood_source}...")
            ood_ss = read_hdf(os.path.join(DATA_DIR, f"{ood_source}_ood_test.h5"))
            ood_ss_cols = [each[1] for each in ood_ss.sources.columns]
            ood_ss_col_idx = ood_ss_cols.index(ood_source)
            ood_contribs = ood_ss.sources.values[:, ood_ss_col_idx]

            ood_lts = ood_ss.info.live_time.values
            ood_cnts = ood_ss.info.total_counts.values
            ood_snrs = np.array(ood_cnts / np.sqrt(ood_lts * STATIC_SYNTH_BG_CPS))

            model.predict(ood_ss, bg_cps=STATIC_SYNTH_BG_CPS)

            fig, ax = generate_ood_heatmap(
                ood_snrs,
                ood_contribs,
                ood_ss.info.ood.values,
                plt_figsize,
                plt_dpi,
                nbins=20
            )
            ood_heatmap_img = wandb.Image(
                fig,
                caption=f"OOD proportion vs. SNR vs. OOD FNR for {ood_source}"
            )
            wandb.log({f"OOD_heatmap_{ood_source}": ood_heatmap_img})
            plt.close()

            fig, ax = generate_recon_error_vs_ood_contribution_plot(
                ood_ss.info[f"unsup_{model.unsup_loss}_loss"].values,
                ood_contribs,
                plt_figsize,
                plt_dpi
            )
            recon_error_vs_ood_contrib_img = wandb.Image(
                fig,
                caption=f"Reconstruction error vs. OOD proportion for {ood_source}"
            )
            wandb.log(
                {f"recon_error_vs_{ood_source}_proportion_plot": recon_error_vs_ood_contrib_img}
            )
            plt.close()

            ood_fnr = 1 - np.mean(ood_ss.info.ood.values)
            wandb.log({
                f"final_ood_fnr_{ood_source}": ood_fnr
            })

        train_fpr = np.mean(train_ss.info.ood.values)
        test_fpr = np.mean(test_ss.info.ood.values)
        wandb.log({
            "final_train_fpr": train_fpr,
            "final_test_fpr": test_fpr
        })

        train_avg_recon_error = np.mean(train_ss.info[f"unsup_{model.unsup_loss}_loss"].values)
        test_avg_recon_error = np.mean(test_ss.info[f"unsup_{model.unsup_loss}_loss"].values)
        wandb.log({
            "final_train_recon_error": train_avg_recon_error,
            "final_test_recon_error": test_avg_recon_error
        })

    wandb.finish()
