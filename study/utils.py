# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Contains functions used during training and for visualization.
"""
import os
from glob import glob

import keras
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from config import (EVAL_LOCAL_BEST_MODELS, EVAL_USE_LOCAL,
                    EVAL_WANDB_BEST_MODELS, MODEL_DIR, MODEL_WANDB_HOST)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from riid.models.neural_nets import LabelProportionEstimator
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score


class SaveTruthsandPredictionsCallback(keras.callbacks.Callback):
    def __init__(self, x_test, y_test, model_name, activation_func,
                 callback_dir: str = "callbacks"):
        self.x_test = x_test
        self.y_test = y_test
        self.model_name = model_name
        self.callback_dir = callback_dir
        self.activation_func = activation_func

        y_test.to_csv(f"{self.callback_dir}/{self.model_name}_truth.csv")

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_test, verbose=0)
        y_pred = self.activation_func(
            tf.convert_to_tensor(y_pred, dtype=tf.float32)
        )
        df = pd.DataFrame(y_pred)

        file_path = os.path.join(self.callback_dir, f"{self.model_name}_{epoch}_pred.csv")
        df.to_csv(file_path)


def animate_calibration(data_path: str, save_path: str, plt_max: float = 0.3):
    # Prepare truth
    truth_file = glob(f"{data_path}/*truth.csv")[0]
    t_df = pd.read_csv(truth_file, skiprows=3, header=None)
    t_df = t_df.drop(0, axis=1)
    mixture_size = t_df.shape[1]
    expected_mean = 1 / mixture_size
    truths = t_df.to_numpy().flatten()

    # Prepare predictions
    pred_files = glob(f"{data_path}/*pred.csv")
    pred_files = {k: v for k, v in zip([int(x.split("_")[-2]) for x in pred_files], pred_files)}
    preds = {}
    for i, f in sorted(pred_files.items()):
        p_df = pd.read_csv(f)
        p_df = p_df.drop("Unnamed: 0", axis=1)
        preds[i] = p_df.to_numpy().flatten()

    # Construct animation
    fig, ax = plt.subplots()
    line1, = ax.plot((0, 1), (0, 1),
                     label="Perfect calibration", color="black",
                     linestyle="dashed")
    line2, = ax.plot((0, 1), (expected_mean, expected_mean),
                     label=f"Mean ({expected_mean:.3f})", color="black",
                     linestyle="dotted")
    legend1 = ax.legend(handles=[line1, line2], loc="upper left")
    ax.add_artist(legend1)
    artists = []
    for i, p in sorted(preds.items()):
        scat = ax.scatter(truths, p, alpha=0.2, color="blue",
                          label=f"epoch {i}")
        scat_legend = ax.legend(handles=[scat], loc="lower right")
        ax.add_artist(scat_legend)
        artists.append([scat, scat_legend])

    ax.set_title(f"Calibration of {mixture_size}-Mixture Proportion Estimator During Training")
    ax.set_xlabel("True Proportion")
    ax.set_ylabel("Predicted Proportion")
    ax.set_xlim((0, plt_max))
    ax.set_ylim((0, plt_max))

    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=100, blit=False)
    ani.save(filename=os.path.join(save_path, "calibration.gif"), writer="pillow")
    ani.save(filename=os.path.join(save_path, "calibration.html"), writer="html")
    plt.close()


def generate_mae_vs_snr_plot(maes, snrs, plt_figsize, plt_dpi, nbins=10):
    plt_snr_steps = np.linspace(
        min(snrs),
        max(snrs),
        nbins+1
    )
    plt_snr_inds = [np.where(
        (snrs > plt_snr_steps[i]) & (snrs <= plt_snr_steps[i+1])
    )[0].astype(int) for i in range(nbins)]
    plt_maes = [maes[each] for each in plt_snr_inds]
    plt_snrs = [np.mean(snrs[each]) for each in plt_snr_inds]

    fig, ax = plt.subplots(figsize=plt_figsize, dpi=plt_dpi)
    ax.boxplot(
        plt_maes,
        positions=plt_snrs,
        vert=1,
        widths=(np.max(snrs) - np.min(snrs))/(2*nbins)
    )
    xticks = np.linspace(
        np.max(snrs),
        np.min(snrs),
        nbins+1
    )
    xticklabels = [f"{each:.0f}" for each in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("SNR")
    ax.set_ylabel("MAE")

    return fig, ax


def generate_calibration_plot(y_true, y_pred, plt_figsize, plt_dpi, plt_max, plt_alpha):
    fig, ax = plt.subplots(figsize=plt_figsize, dpi=plt_dpi)
    ax.plot(
        np.linspace(0, plt_max, 1000),
        np.linspace(0, plt_max, 1000),
        label="Ideal Calibration",
        linestyle="--",
        color="black"
    )
    ax.scatter(
        y_true,
        y_pred,
        label=f"Sample (MAE = {mae(y_true, y_pred):.5f}, r^2 = {r2_score(y_true, y_pred):.3f})",
        alpha=plt_alpha
    )
    ax.set_xlim((0, plt_max))
    ax.set_ylim((0, plt_max))
    ax.legend()
    ax.set_ylabel("Predicted Proportion")
    ax.set_xlabel("True Proportion")

    return fig, ax


def generate_threshold_plot(recon_errors, snrs, threshold_func, plt_figsize, plt_dpi, plt_alpha):
    fig, ax = plt.subplots(figsize=plt_figsize, dpi=plt_dpi)
    ax.scatter(snrs, recon_errors, alpha=plt_alpha, label="synthetic training sample")
    x = np.linspace(np.min(snrs), np.max(snrs), 100)
    ax.plot(x, threshold_func(x), linestyle="--", label="U-spline threshold func", color="red")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Reconstruction Error")

    return fig, ax


def generate_ood_heatmap(snrs, ood_contribs, ood_decisions, plt_figsize, plt_dpi, nbins=20):
    fns = np.zeros((nbins, nbins))
    snr_steps = np.linspace(
        np.min(snrs),
        np.max(snrs),
        nbins+1
    )
    ood_prop_steps = np.linspace(
        0.0,
        1.0,
        nbins+1
    )

    for i in range(nbins):
        for j in range(nbins):
            snr_inds = np.where(
                (snrs > snr_steps[i]) &
                (snrs <= snr_steps[i+1])
            )
            prop_inds = np.where(
                (ood_contribs[snr_inds[0]] > ood_prop_steps[j]) &
                (ood_contribs[snr_inds[0]] <= ood_prop_steps[j+1])
            )

            fns[i, j] = 1.0 - ood_decisions[snr_inds[0]][prop_inds[0]].mean()

    fns = pd.DataFrame(
        fns
    ).interpolate().values

    fig, ax = plt.subplots(figsize=plt_figsize, dpi=plt_dpi)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(
        fns.T,
        extent=[np.min(snrs), np.max(snrs), 0, 1.0],
        origin="lower",
        aspect="auto",
        interpolation="nearest"
    )
    ax.set_xlabel("SNR")
    ax.set_ylabel("OOD Proportion")
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_label("OOD FNR")
    cb.ax.set_ylim(0.0, 1.0)
    fig.tight_layout()

    return fig, ax


def generate_recon_error_vs_ood_contribution_plot(recon_errors, ood_contribs,
                                                  plt_figsize, plt_dpi, nbins=10):
    ood_contrib_steps = np.linspace(
        0.0,
        1.0,
        nbins+1
    )
    recon_error_inds = [np.where(
        (ood_contribs > ood_contrib_steps[i]) & (ood_contribs <= ood_contrib_steps[i+1])
    )[0].astype(int) for i in range(nbins)]
    plt_recon_errors = [recon_errors[each] for each in recon_error_inds]
    plt_ood_contribs = [np.mean(ood_contribs[each]) for each in recon_error_inds]

    fig, ax = plt.subplots(figsize=plt_figsize, dpi=plt_dpi)
    ax.boxplot(
        plt_recon_errors,
        positions=plt_ood_contribs,
        vert=1,
        widths=0.05
    )
    xticks = np.linspace(
        0.0,
        1.0,
        6
    )
    xticklabels = [f"{each:.2f}" for each in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("OOD Proportion")
    ax.set_ylabel("Reconstruction Error")

    return fig, ax


def load_final_models():
    if EVAL_USE_LOCAL:
        best_runs = EVAL_LOCAL_BEST_MODELS
        best_models = []

        for unsup_loss in best_runs.keys():
            model_filename = best_runs[unsup_loss]
            model = LabelProportionEstimator()
            model.load(os.path.join(MODEL_DIR, model_filename))
            best_models.append(model)

    else:
        best_runs = EVAL_WANDB_BEST_MODELS
        best_models = []

        wandb.login(host=MODEL_WANDB_HOST)

        for unsup_loss in best_runs.keys():
            api = wandb.Api()
            run = api.run(best_runs[unsup_loss])

            filenames = [each.name for each in run.files()]
            model_filename_idx = [
                i for i in range(len(filenames)) if filenames[i].endswith("onnx")
            ][0]
            model_filename = filenames[model_filename_idx]
            model_info_filename_idx = filenames.index(
                os.path.splitext(model_filename)[0] + "_info.json"
            )
            print(model_filename)
            run.files()[model_filename_idx].download(
                root=MODEL_DIR,
                replace=True,
                exist_ok=True
            )
            run.files()[model_info_filename_idx].download(
                root=MODEL_DIR,
                replace=True,
                exist_ok=True
            )

            model = LabelProportionEstimator()
            model.load(os.path.join(MODEL_DIR, model_filename))
            best_models.append(model)

    return best_runs, best_models
