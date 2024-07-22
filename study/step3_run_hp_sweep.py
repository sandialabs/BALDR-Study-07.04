# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Runs a hyperparameter sweep of LPE models.
"""
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from config import (DATA_DIR, FG_SEED_FILE, HP_SWEEP_EPOCHS,
                    HP_SWEEP_LR_SCHED_PATIENCE,
                    HP_SWEEP_LR_SCHED_PATIENCE_MIN_DELTA, HP_SWEEP_PATIENCE,
                    HP_SWEEP_PATIENCE_MIN_DELTA, HP_SWEEP_TRIALS,
                    MIX_TEST_FG_SAMPLES, MIX_TRAIN_BG_ALPHA,
                    MIX_TRAIN_BG_SAMPLES, MIX_TRAIN_BG_SIZE,
                    MIX_TRAIN_FG_LAMBDA, MIX_TRAIN_FG_SAMPLES,
                    MODEL_ACTIVITY_L1_REG, MODEL_ACTIVITY_L2_REG,
                    MODEL_KERNEL_L1_REG, MODEL_KERNEL_L2_REG,
                    MODEL_NORMALIZE_SUP_LOSS, MODEL_OOD_FPR, MODEL_OPTIMIZER,
                    MODEL_SPLINE_BINS, MODEL_SPLINE_K, MODEL_SPLINE_S,
                    MODEL_SUP_LOSS, MODEL_SUP_NORMALIZE_SCALER,
                    MODEL_TARGET_LEVEL, MODEL_TRAIN_ON_GPU,
                    MODEL_TRAIN_ON_STATIC_SYNTH_DATA, MODEL_VAL_SPLIT,
                    MODEL_WANDB_HOST, MODEL_WANDB_PROJECT, OOD_FG_MIX_SAMPLES,
                    OOD_PROP_RANGE, OOD_PROP_SAMPLES, OOD_SEED_FILE, OOD_SPS,
                    STATIC_SYNTH_BG_CPS, STATIC_SYNTH_LIVE_TIME_RANGE,
                    STATIC_SYNTH_LIVE_TIME_SAMPLING, STATIC_SYNTH_LONG_BG_CPS,
                    STATIC_SYNTH_SNR_RANGE, STATIC_SYNTH_SNR_SAMPLING,
                    STATIC_SYNTH_SPS, STATIC_SYNTH_TEST_SNR_RANGE, TARGET_BINS,
                    TARGET_ECAL, TEST_FG_MIX_FILE, TEST_FILE,
                    TRAIN_FG_MIX_FILE, TRAIN_FILE, TUNING_VAL_FG_MIX_FILE,
                    TUNING_VAL_FILE)
from riid.data.sampleset import read_hdf
from riid.models.neural_nets import LabelProportionEstimator
from sklearn.metrics import mean_absolute_error as mae
from utils import (SaveTruthsandPredictionsCallback, animate_calibration,
                   generate_calibration_plot, generate_mae_vs_snr_plot,
                   generate_ood_heatmap,
                   generate_recon_error_vs_ood_contribution_plot,
                   generate_threshold_plot)
from wandb.keras import WandbMetricsLogger

# Set the unsupervised loss to optimize for this sweep
# Should rerun this script for all four unsupervised losses
# ("chi_squared", "jsd", "poisson_nll", "sse")
unsup_loss = "chi_squared"

# Try to handly pesky warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)

# Load in data
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

if MODEL_TRAIN_ON_STATIC_SYNTH_DATA:
    train_ss = read_hdf(TRAIN_FILE)
    test_ss = read_hdf(TEST_FILE)
    val_ss = read_hdf(TUNING_VAL_FILE)
else:
    train_ss = read_hdf(TRAIN_FG_MIX_FILE)
    test_ss = read_hdf(TEST_FG_MIX_FILE)
    val_ss = read_hdf(TUNING_VAL_FG_MIX_FILE)

# Enable GPU
if MODEL_TRAIN_ON_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # optionally select valid gpus
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

wandb.login(host=MODEL_WANDB_HOST)

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "final_tuning_val_mae",
        "goal": "minimize"
    },
    "name": f"sweep_{unsup_loss}"
}

sweep_parameters = {
    "preprocessing_spectra_bins": {"value": TARGET_BINS},
    "preprocessing_spectra_ecal": {"value": TARGET_ECAL},
    "SeedMixer_train_bg_size": {"value": MIX_TRAIN_BG_SIZE},
    "SeedMixer_train_bg_alpha": {"value": MIX_TRAIN_BG_ALPHA},
    "SeedMixer_train_bg_samples": {"value": MIX_TRAIN_BG_SAMPLES},
    "SeedMixer_train_fg_samples": {"value": MIX_TRAIN_FG_SAMPLES},
    "SeedMixer_train_fg_lambda": {"value": MIX_TRAIN_FG_LAMBDA},
    "SeedMixer_test_fg_samples": {"value": MIX_TEST_FG_SAMPLES},
    "StaticSynth_sps": {"value": STATIC_SYNTH_SPS},
    "StaticSynth_bg_cps": {"value": STATIC_SYNTH_BG_CPS},
    "StaticSynth_long_bg_cps": {"value": STATIC_SYNTH_LONG_BG_CPS},
    "StaticSynth_live_time_range": {"value": STATIC_SYNTH_LIVE_TIME_RANGE},
    "StaticSynth_live_time_sampling": {"value": STATIC_SYNTH_LIVE_TIME_SAMPLING},
    "StaticSynth_snr_range": {"value": STATIC_SYNTH_SNR_RANGE},
    "StaticSynth_snr_sampling": {"value": STATIC_SYNTH_SNR_SAMPLING},
    "StaticSynth_snr_test_range": {"value": STATIC_SYNTH_TEST_SNR_RANGE},
    "OOD_prop_range": {"value": OOD_PROP_RANGE},
    "OOD_prop_samples": {"value": OOD_PROP_SAMPLES},
    "OOD_fg_mix_smples": {"value": OOD_FG_MIX_SAMPLES},
    "OOD_sps": {"value": OOD_SPS},
    "model_train_on_static_synth_data": {"value": MODEL_TRAIN_ON_STATIC_SYNTH_DATA},
    "model_train_on_gpu": {"value": MODEL_TRAIN_ON_GPU},
    "model_target_level": {"value": MODEL_TARGET_LEVEL},
    "model_beta": {
        "distribution": "log_uniform_values",
        "min": 1e-12,
        "max": 1.0
    },
    "model_unsup_loss": {"value": unsup_loss},
    "model_sup_loss": {"value": MODEL_SUP_LOSS},
    "model_optimizer": {"value": MODEL_OPTIMIZER},
    "model_optimizer_epsilon": {
        "distribution": "log_uniform_values",
        "min": 1e-6,
        "max": 1.0
    },
    "model_normalize_sup_loss": {"value": MODEL_NORMALIZE_SUP_LOSS},
    "model_normalize_sup_loss_scaler": {"value": MODEL_SUP_NORMALIZE_SCALER},
    "model_init_lr": {
        "distribution": "log_uniform_values",
        "min": 1e-6,
        "max": 1.0
    },
    "model_batch_size": {
        "distribution": "q_uniform",
        "q": 50,
        "min": 100,
        "max": 500,
    },
    "model_epochs": {"value": HP_SWEEP_EPOCHS},
    "model_hidden_layer_1": {
        "distribution": "q_uniform",
        "min": 256,
        "max": 1024,
    },
    "model_hidden_layer_2": {
        "distribution": "q_uniform",
        "min": 256,
        "max": 1024,
    },
    "model_hidden_layer_activation": {
        "values": ["relu", "mish", "elu", "tanh"]
    },
    "model_kernel_l1_reg": {"value": MODEL_KERNEL_L1_REG},
    "model_kernel_l2_reg": {"value": MODEL_KERNEL_L2_REG},
    "model_activity_l1_reg": {"value": MODEL_ACTIVITY_L1_REG},
    "model_activity_l2_reg": {"value": MODEL_ACTIVITY_L2_REG},
    "model_dropout": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.5
    },
    "model_patience": {"value": HP_SWEEP_PATIENCE},
    "model_patience_delta": {"value": HP_SWEEP_PATIENCE_MIN_DELTA},
    "model_lr_patience": {"value": HP_SWEEP_LR_SCHED_PATIENCE},
    "model_lr_patience_delta": {"value": HP_SWEEP_LR_SCHED_PATIENCE_MIN_DELTA},
    "model_val_split": {"value": MODEL_VAL_SPLIT},
    "model_ood_fpr": {"value": MODEL_OOD_FPR},
    "model_spline_bins": {"value": MODEL_SPLINE_BINS},
    "model_spline_k": {"value": MODEL_SPLINE_K},
    "model_spline_s": {"value": MODEL_SPLINE_S}
}

sweep_config["parameters"] = sweep_parameters

sweep_id = wandb.sweep(sweep=sweep_config, project=MODEL_WANDB_PROJECT)


def train(config=sweep_config):
    with wandb.init(config=config):
        config = wandb.config

        time_str = time.strftime("%Y%m%d-%H%M%S")
        model_name = f"lpe_{MODEL_SUP_LOSS}_{unsup_loss}_{config.model_beta}_{time_str}"

        run_dir = wandb.run.dir
        print(f"run directory: {run_dir}")
        callback_dir = os.path.join(run_dir, "callbacks/")
        if not os.path.exists(callback_dir):
            os.makedirs(callback_dir)

        model = LabelProportionEstimator(
            hidden_layers=(config.model_hidden_layer_1, config.model_hidden_layer_2),
            sup_loss=config.model_sup_loss,
            unsup_loss=config.model_unsup_loss,
            beta=config.model_beta,
            fg_dict=None,
            optimizer=config.model_optimizer,
            optimizer_kwargs={"epsilon": config.model_optimizer_epsilon},
            learning_rate=config.model_init_lr,
            metrics=["mae"],
            hidden_layer_activation=config.model_hidden_layer_activation,
            kernel_l1_regularization=config.model_kernel_l1_reg,
            kernel_l2_regularization=config.model_kernel_l2_reg,
            activity_l1_regularization=config.model_activity_l1_reg,
            activity_l2_regularization=config.model_activity_l2_reg,
            dropout=config.model_dropout,
            target_level=config.model_target_level,
            bg_cps=config.StaticSynth_bg_cps,
            fit_spline=config.model_train_on_static_synth_data,
            ood_fp_rate=config.model_ood_fpr,
            spline_bins=config.model_spline_bins,
            spline_k=config.model_spline_k,
            spline_s=config.model.spline_s
        )

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=HP_SWEEP_LR_SCHED_PATIENCE,
                min_delta=HP_SWEEP_LR_SCHED_PATIENCE_MIN_DELTA
            ),
            SaveTruthsandPredictionsCallback(
                    test_ss.spectra,
                    test_ss.sources,
                    model_name,
                    model.activation,
                    callback_dir=callback_dir
            ),
            WandbMetricsLogger(log_freq="epoch"),
        ]

        _ = model.fit(
            seeds_ss=fg_seeds_ss,
            ss=train_ss,
            batch_size=config.model_batch_size,
            epochs=HP_SWEEP_EPOCHS,
            validation_split=config.model_val_split,
            callbacks=callbacks,
            patience=HP_SWEEP_PATIENCE,
            verbose=True,
            normalize_scaler=config.model_normalize_sup_loss_scaler,
            normalize_sup_loss=config.model_normalize_sup_loss,
            bg_cps=config.StaticSynth_bg_cps,
            es_min_delta=HP_SWEEP_PATIENCE_MIN_DELTA
        )

        model.save(os.path.join(run_dir, f"{model_name}.onnx"))

        print("Running forward pass on train/test/val data...")
        model.predict(train_ss, bg_cps=config.StaticSynth_bg_cps)
        train_true_props = train_ss.sources.values.flatten()
        train_pred_props = train_ss.prediction_probas.values.flatten()
        train_mae = mae(train_true_props, train_pred_props)

        model.predict(test_ss, bg_cps=config.StaticSynth_bg_cps)
        test_true_props = test_ss.sources.values.flatten()
        test_pred_props = test_ss.prediction_probas.values.flatten()
        test_mae = mae(test_true_props, test_pred_props)

        model.predict(val_ss, bg_cps=config.StaticSynth_bg_cps)
        val_true_props = val_ss.sources.values.flatten()
        val_pred_props = val_ss.prediction_probas.values.flatten()
        val_mae = mae(val_true_props, val_pred_props)

        wandb.log({
            "final_train_mae": train_mae,
            "final_test_mae": test_mae,
            "final_tuning_val_mae": val_mae
        })

        print("Generating calibration plots for train/test/val data..")
        plt_max = np.max(np.concatenate((
            train_true_props,
            train_pred_props,
            test_true_props,
            test_pred_props,
            val_true_props,
            val_pred_props
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

        fig, ax = generate_calibration_plot(
            val_true_props,
            val_pred_props,
            plt_figsize,
            plt_dpi,
            plt_max,
            plt_alpha)

        val_calibration_img = wandb.Image(
            fig,
            caption="calibration plot for tuning validation dataset"
        )
        wandb.log({"tuning_val_calibration_plot": val_calibration_img})
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

        # delete callback directory to save space
        shutil.rmtree(callback_dir)

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

            val_maes = np.array([
                mae(
                    val_ss.sources.values[i, :],
                    val_ss.prediction_probas.values[i, :]
                ) for i in range(val_ss.n_samples)
            ])
            val_lts = val_ss.info.live_time.values
            val_cnts = val_ss.info.total_counts.values
            val_snrs = np.array(val_cnts / np.sqrt(val_lts * STATIC_SYNTH_BG_CPS))

            print("Generating mae vs. snr plots for train/test/val data...")
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

            fig, ax = generate_mae_vs_snr_plot(
                val_maes,
                val_snrs,
                plt_figsize,
                plt_dpi
            )
            val_mae_vs_snr_img = wandb.Image(
                fig,
                caption="mae vs. snr for tuning validation dataset"
            )
            wandb.log({"tuning_val_mae_vs_snr_plot": val_mae_vs_snr_img})
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
            ood_fnrs = []
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
                wandb.log({
                    f"recon_error_vs_{ood_source}_proportion_plot":
                    recon_error_vs_ood_contrib_img}
                )
                plt.close()

                ood_fnr = 1 - np.mean(ood_ss.info.ood.values)
                wandb.log({
                    f"final_ood_fnr_{ood_source}": ood_fnr
                })
                ood_fnrs.append(ood_fnr)

            train_fpr = np.mean(train_ss.info.ood.values)
            test_fpr = np.mean(test_ss.info.ood.values)
            val_fpr = np.mean(val_ss.info.ood.values)
            wandb.log({
                "final_train_fpr": train_fpr,
                "final_test_fpr": test_fpr,
                "final_tuning_val_fpr": val_fpr,
                "final_test_ood_fnr": np.mean(ood_fnrs)
            })

            train_avg_recon_error = np.mean(
                train_ss.info[f"unsup_{model.unsup_loss}_loss"].values
            )
            test_avg_recon_error = np.mean(
                test_ss.info[f"unsup_{model.unsup_loss}_loss"].values
            )
            val_avg_recon_error = np.mean(
                val_ss.info[f"unsup_{model.unsup_loss}_loss"].values
            )
            wandb.log({
                "final_train_recon_error": train_avg_recon_error,
                "final_test_recon_error": test_avg_recon_error,
                "final_tuning_val_recon_error": val_avg_recon_error
            })


wandb.agent(sweep_id, function=train, count=HP_SWEEP_TRIALS)

wandb.finish()
