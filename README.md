---
title: README
author:
- Alan Van Omen (ajvanom@sandia.gov)
- Tyler Morrow (tmorro@sandia.gov)
geometry: margin=1in
---
This repository contains the code and synthetic data used to produce the results for the paper
*Multilabel Proportion Prediction and Out-of-Distribution Detection on Gamma Spectra of Short-Lived Fission Products*.


# Acknowledgements

This work was performed by employees of Sandia National Laboratories and funded by the U.S. Department of Energy, National Nuclear Security Administration, Office of Defense Nuclear Nonproliferation Research and Development (DNN R&D).

Sandia National Laboratories is a multimission laboratory managed and operated by National Technology \& Engineering Solutions of Sandia, LLC, a wholly owned subsidiary of Honeywell International Inc., for the U.S. Department of Energy's National Nuclear Security Administration under contract DE-NA0003525.

Contributions from C. Scott were supported in part by the National Science Foundation under award 2008074, and by the Department of Defense, Defense Threat Reduction Agency under award HDTRA1-20-2-0002.


# Manifest

## `code/`

This directory contains the code used to generate data, models, and results.
It also contains the initial data files necessary to generate the synthetic training/testing data, as well as the best models found the in the paper.
There is a `requirements.txt` file which contains a list of the necessary packages you will need to install in your Python environment (tested with Python 3.9.6 on macOS Ventura and Red Hat Linux 7.9).

```sh
pip install -r requirements.txt
```

The code files we intend for you to run begin with "step#" where "#" indicates the execution order.
Doc-strings in each code file describe their purpose.
The final results are generated in Jupyter notebooks which use the data and models to produce results.
The `config.py` file is referenced by all steps and centralizes nearly all critical parameters.

Also note that the following steps require access to a Weights & Biases developer platform [\[1\]][1]:

- `code/step3_run_hp_sweep.py`
- `code/step5_find_best_model.ipynb`

**Skip these steps if you do not have access to such an instance.**
By default, the code assumes you do not have a W&B instance, which enables you to run step 4 locally and at least train a model.
However, a W&B instance is one requirement for fully recreating the study results.
If you do have a W&B instance, make sure to configure the `MODEL_USE_WANDB`, `MODEL_WANDB_HOST`, `MODEL_WANDB_PROJECT` parameters in `config.py`.

Also, you can choose to either evaluate local models (which you either trained locally or downloaded from Weights & Biases) or automatically download models from Weights & Biases to evaluate.
This is controlled via the `EVAL_USE_LOCAL` parameter in `code/config.py`.
The initial values you find in `config.py` are for the best models found in our study and described in the paper.
However, if you choose to automatically download models from a Weights & Biases run, be sure to fill in the path to each run in the `EVAL_WANDB_BEST_MODELS` parameter.


### `code/data/`

This directory contains the synthetic seed files obtained via GADRAS injects which are used to generate all the synthetic data for the paper.

- `code/data/U-235_th_FALCON_11MIN_bg_seeds.h5` contains signatures for K, U, Th, and cosmic radiation which were used to generate representative background mixtures.
- `code/data/U-235_th_FALCON_11MIN_fg_seeds.h5` contains signatures for all the fission products identified in the SME analysis of our target spectrum.
- `code/data/U-235_th_FALCON_11MIN_ood_seeds.h5` contains signatures for Ce134, Cs137, and Mo99 which were used to simulate out-of-distribution (OOD) sources.

All seed files were obtained via inject with GADRAS 19.2.3 using a detector response function (DRF) based on a Falcon 5000 detector.


#### PNNL Data

In order to obtain access to the measured data from PNNL, readers will have to reach out to the authors of [\[3\]][3].
Once obtained, place the data in `data/pnnl/` preserving the original file structure.
Next, you will need to convert all `.n42` files to `.pcf`.
You can process an entire directory structure in this way on the command line using Cambio [\[3\]][3].
Note: do not have the `.pcf` files output to a separate directory, leave them next to the existing `.n42` files (read `cambio --help` for more).


### `code/models`

This directory is the destination for any trained model. If you do not want to perform synthesis and train a model, we have included the pre-trained "best" models from the paper (one for each unsupervised loss function) which were used to generate the final results.
The model file is in the Open Neural Network Exchange (ONNX) format and is accompanied by a JSON file providing additional, describing metadata [\[4\]][4].
If you are re-running the experiment, note that this folder is where models are output, so you will see many more appear.

The ONNX file will require the ONNX runtime to load and execute; it is primarily a binary file of sorts and not meant to be viewed. Its accompanying JSON file can be viewed by your text editor of choice.


### `code/runs`

If you do not use Weights and Biases, then the directory `code/runs` will be created and will locally store all the trained ONNX models along with associated run files.
If you use Weights and Biases, models will be stored on the cloud by default.


# Methodology

At a high level, the code represents the following process to conduct the study outlined in our paper.

- The provided foreground and background seeds are randomly mixed in PyRIID using the `SeedMixer`.
- The mixed seeds where used to generate noisy gamma spectrum which vary in terms of signal-to-noise ratio using the PyRIID `StaticSynthesizer`.
- Training data from the previous step was then used to fit many models. Note that training consisted of hyperparameter tuning which looked for optimal parameters, and then a final step which trained an "optimal" model.
- With a model found, the test data was then used in several notebooks to generate the final results, mainly plots.


# References

1. [Weights & Biases][1]
2. [Cambio][2]
3. [Methods to Collect, Compile, and Analyze Observed Short-lived Fission Product Gamma Data][3]
4. [ONNX][4]

[1]: https://wandb.ai/site
[2]: https://doi.org/10.11578/dc.20210416.65
[3]: https://www.osti.gov/biblio/1028568
[4]: https://github.com/onnx/onnx
