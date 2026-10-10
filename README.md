# Probabilistic Forecasting for Anomaly Detection

In this project, we propose a new framework for anomaly detection in irregularly sampled multi-variate time-series and apply it to synthetic data and infant gut microbiome trajectories. The method is based on neural jump ordinary differential equations (NJODEs) and infers conditional mean and variance trajectories in a fully path dependent way to compute anomaly scores. Read more about this study in our preprint [here](https://arxiv.org/abs/2510.00087):

Adamov, A., Chardonnet, M., Krach, F., Heiss, J., Teichmann, J., Bokulich, N. A. Revealing the Temporal Dynamics of Antibiotic Anomalies in the Infant Gut Microbiome with Neural Jump ODEs. 2025. doi: 
https://doi.org/10.48550/arXiv.2510.00087


## Dependencies

The project uses [uv](https://docs.astral.sh/uv/getting-started/installation/)
to manage Python and dependencies. Python 3.12 is the recommended version
(the project requires Python 3.12 or newer).

Install uv, then run the following from the repository root:

```bash
uv python install 3.12
uv sync --locked --python 3.12
```

This creates the local `.venv` and installs the versions recorded in `uv.lock`.
Dependencies are declared in `pyproject.toml`.
Git is required to install the Telegram notifications dependency from GitHub.
On macOS, compiling `iisignature` also requires the Xcode command-line tools
(`xcode-select --install` if they are not already installed).

Use `uv run` to run commands in the project environment without
activating it. Run the data generation, training, and evaluation scripts from
`src`, since their data paths are relative to that directory:

```bash
cd src
uv run python run.py --help
```

For PyCharm or another IDE, select the repository's `.venv/bin/python` as the
interpreter. To add or change dependencies, use `uv add <package>` or
`uv remove <package>` from the repository root and commit both
`pyproject.toml` and `uv.lock`.

### Telegram notifications

For commands with `--SEND=True`, put the bot token on a single line in
`.telegram-token` at the repository root. This file is ignored by Git.
On macOS/Linux, restrict access with `chmod 600 .telegram-token`.
`src/configs/config.py` loads it and supplies it to all notification calls;
configure `CHAT_ID` and `ERROR_CHAT_ID` there for the destination chats.
Use `--SEND=False` when notifications are not needed.

---
# Synthetic Data with ingested anomalies
## Configurations 
- The configurations can be accessed through the following folder :
    ```bash
    cd src/configs
    ```
- Those are the configurations for generating the synthetic data, training the forecasting and anomaly detection models, plotting and monitoring.

## Generating the data

  - Generation of synthetic data from the modified Orstein-Uhlenbeck process without anomalies (base model) for training and evaluation: 
    ```bash
    cd src
    uv run python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_3_dict --seed=0
    uv run python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_eval --seed=1
    ```

  - Generation of synthetic data with anomalies:
      ```bash
      cd src
      uv run python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_diffusion_dict --seed=0
      uv run python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_diffusion_dict_eval --seed=1
      uv run python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_noise_dict --seed=0
      uv run python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_noise_dict_eval --seed=1
      uv run python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_cutoff_dict --seed=0
      uv run python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_cutoff_dict_eval --seed=1
      uv run python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_spike_dict --seed=0
      uv run python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_spike_dict_eval --seed=1
      ```
  - Generation of plots of base and anomaly datasets:
    ```bash
    uv run python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_plot --seed=0 --plot_paths=[0]
    uv run python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_diffusion_dict_plot --seed=0 --plot_paths=[0]
    uv run python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_noise_dict_plot --seed=0 --plot_paths=[0]
    uv run python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_cutoff_dict_plot --seed=0 --plot_paths=[0]
    uv run python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_spike_dict_plot --seed=0 --plot_paths=[0]
    ```


## Training the probabilistic forecasting module

Important flags:

- **params**: name of the params list (defined in config.py) to use for parallel run
- **NB_JOBS**: nb of parallel jobs to run with joblib
- **USE_GPU**: whether to use GPU for training
- **NB_CPUS**: nb of CPUs used by each training
- **saved_models_path**: path where the models are saved

- Training with PD-NJODE on the sythetic data :
    ```bash
    cd src
    uv run python run.py --params=param_list_AD_OrnsteinUhlenbeckWithSeason_3 --NB_JOBS=1 --NB_CPUS=1 --USE_GPU=False --get_overview=overview_dict_AD_OrnsteinUhlenbeckWithSeason
    ```


## Training / Evaluating the Anomaly detection modules on data with ingested anomalies

Important flags:

- **ad_params**: name of the anomaly detection params list (defined in config.py)
- **forecast_saved_models_path**: path where the forecast models are saved
- **forecast_model_ids**: List of forecast model ids to run

- Training and Evaluating the Anomaly Detection Module on synthetic data :
    ```bash
    cd src
    uv run python evaluate_AD.py --forecast_model_ids=[1] --forecast_saved_models_path=../data/saved_models_AD_OrnsteinUhlenbeckWithSeason/ --ad_params=param_dict_AD_modules
    # plot from the last (or best) AD model
    uv run python evaluate_AD.py --plot_only=True --quant_eval_in_plot_only=True --which_AD_model_load="last" --forecast_model_ids=[1] --forecast_saved_models_path=../data/saved_models_AD_OrnsteinUhlenbeckWithSeason/ --ad_params=param_dict_AD_modules
    ```

---
# Anomaly Detection in Microbiome Data
## Generate Dataset:

novel alpha diversity metric datasets:
```shell
# the following is the dataset which is used for the final evaluation
uv run python make_microbial_dataset.py --dataset_config=config_novel_alpha_faith_pd
uv run python make_microbial_dataset.py --dataset_config=config_novel_alpha_faith_pd_w_geo

uv run python make_microbial_dataset.py --dataset_config=config_entero_family
uv run python make_microbial_dataset.py --dataset_config=config_entero_family_w_geo
uv run python make_microbial_dataset.py --dataset_config=config_entero_genus
uv run python make_microbial_dataset.py --dataset_config=config_entero_genus_w_geo

uv run python make_microbial_dataset.py --dataset_config=config_novel_alpha_faith_pd_entero_family
uv run python make_microbial_dataset.py --dataset_config=config_novel_alpha_faith_pd_entero_genus
uv run python make_microbial_dataset.py --dataset_config=config_novel_alpha_faith_pd_entero_genus_scaled
```


## Training PD-NJODE:
novel alpha diversity metric datasets:
```shell
uv run python run.py --params=param_list_microbial_novel_alpha_div --NB_JOBS=24 --NB_CPUS=1 --SEND=True --USE_GPU=False --first_id=1 --get_overview=overview_dict_microbial_novel_alpha_div

# the following is the model training which is used for the final evaluation
uv run python run.py --params=param_list_microbial_novel_alpha_div2 --NB_JOBS=24 --NB_CPUS=1 --SEND=True --USE_GPU=False --first_id=1 --get_overview=overview_dict_microbial_novel_alpha_div2
uv run python run.py --plot_paths=plot_paths_novel_alpha_div2
```

Plot the conditional mean and uncertainty bands for **all antibiotic-exposed
infants from the AD train and validation splits** using model 57's best
checkpoint and the uv environment:

```shell
cd src
uv run python run.py --plot_paths=plot_paths_novel_alpha_div2_abx
```

This selects exposed trajectories from the union of `all/train_idx.npy` and
`all/val_idx.npy`, retaining the
checkpoint's `no_abx` training split. It verifies that training has no observed
exposure flags or positive exposure counts, and that selected exposed rows
do not overlap training. In the current `microbial_novel_alpha_faith_pd` dataset,
model 57 has 112 training trajectories with no recorded post-exposure
observations, and all 141 exposed trajectories (112 from AD train and 29 from
AD validation) are plotted. These exposed infants were excluded from model 57's
forecasting training. Other training
splits, including `all`, contain exposed infants. These checks concern recorded
exposures; they do not establish that an infant never received antibiotics
outside the observed history.

Points are red when their cumulative `abx_exposure` count exceeds the count at
the previous real observation. The first real observation is compared with
zero; missing observation times and the filled time-zero value are skipped.
Red marks the first measurement showing a new exposure, rather than its exact
administration time. Other observations use the usual color. Predictions
condition on available observations throughout the trajectory; these are
updated trajectories, not forecasts frozen before exposure or estimates of a
causal antibiotic effect.

PDFs are saved in
`data/saved_models_microbial_novel_alpha_div2/id-57/plots/abx_trajectories/`
as `abx-plot_epoch-<epoch>_sample-<original_dataset_row>_host-<infant_id>.pdf`.
The sample identifier is the original zero-based dataset row, not its position
within the validation batch; the host identifier identifies the infant.

To add a matching anomaly-score panel and stop microbial observation updates
at the selected antibiotic exposure, run the AD plotting option from `src`:

```shell
uv run python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --ad_params=param_list_AD_microbial_novel_alpha_div2 --plot_trajectories=True
```

This computes and saves the usual AD scores and produces 141 two-panel plots
for each `only_jump_before_abx_exposure` value in the AD configuration (currently
1, 2, and 3). Microbial observations with cumulative exposure counts at or above
the threshold no longer update the model. Dynamic covariates continue to update
when `use_dyn_cov_after_abx=True`, matching the standard AD settings. Infants
who never reach a particular threshold retain all their observation updates.

The upper panel shows the conditional mean and mean ± 1.96 times the scoring
standard deviation, using the same variance corrections and scaling factors as
the normal left-tail score in the lower panel. Scores appear at actual
observation times, with new exposures colored red in both panels. Dashed lines
mark the first observation reaching the exposure threshold, not the exact
administration date. The horizontal axis is infant age in days.

Outputs are grouped under
`data/saved_models_microbial_novel_alpha_div2/id-57/plots/abx_trajectories/only_jump_before_abx_exposure-<value>/`,
then by checkpoint, distribution, tail, coordinate, and scaling settings.
Filenames preserve original dataset row and infant IDs. The scaling-factor CSVs
must already exist for the standard scaled AD configuration; see the score
computation instructions below. The earlier `run.py` command remains available
for unrestricted updated trajectories without a score panel.

Based on the training results, the model ids 55,56,57 (using RNN, no signature and increasing probability to only use dynamic features as inputs; with the 3 different validation horizons) are selected. Computing the outputs on the validation set shows that only 57 never predicts a cond variance <0. 
Moreover, this model produces increasing confidence interval sizes and it has the best eval loss (i.e. train loss on val set) of the 3 models. Therefore, this model is selected for further evaluation.


## Compute Anomaly Detection Scores:

novel alpha diversity metric datasets with scaling factors:
WARNING: the following things need to be run in the given order to work properly
```shell
# first compute the z-scores and the scaling factors
uv run python Microbial_AD_eval.py --NB_CPUS=1 --forecast_model_ids=AD_microbial_novel_alpha_div_ids2 --ad_params=param_list_AD_microbial_novel_alpha_div2_scaling_factors --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False
uv run python Microbial_AD_eval.py --NB_CPUS=1 --forecast_model_ids=AD_microbial_novel_alpha_div_ids2 --ad_params=param_list_AD_microbial_novel_alpha_div2_scaling_factors2 --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=False --evaluate_scores=False --compute_zscore_scaling_factors=True

# then compute the AD scores using the scaling factors
uv run python Microbial_AD_eval.py --NB_CPUS=1 --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2 --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False

# and do a preliminary evaluation of them (only works if AD scores have been computed for only_jump_before_abx_exposure=1,2,3 before)
uv run python Microbial_AD_eval.py --NB_CPUS=1 --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_ev --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=False --evaluate_scores=True --compute_zscore_scaling_factors=False

# then compute scores for reliability evaluation using the scaling factors
uv run python Microbial_AD_eval.py --NB_CPUS=1 --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_reliability_eval --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False

# as reference, compute scores without scaling factors
uv run python Microbial_AD_eval.py --NB_CPUS=1 --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_nsf --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False
uv run python Microbial_AD_eval.py --NB_CPUS=1 --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_reliability_eval_nsf --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False
```


## Score evaluation
Once you trained a microbiome-based model, you can evaluate the resulting scores in the notebook `results/evaluate_scores.ipynb`. To run the notebook, create and activate this conda environment:
````
conda create --name score_eval numpy pandas matplotlib seaborn scipy ipython ipykernel -y
conda activate score_eval
pip install -e .
````
With the same conda environment, a score time horizon reliability analysis can be performed that evaluates for how long the predicted scores remain constant on the no antibiotics scores (notebook `results/analyse_t_horizon_reliability.ipynb`) 
and the predictive performance of the inferred anomaly scores can be compared to baseline predictions in notebook `results/evaluate_predictions.ipynb`. 
Also, resulting anomaly scores can be compared to matched alpha diversity values in notebook `evaluate_matched_alpha.ipynb` 
and individual score trajectories can be explored in `evaluate_indiv_score_increase.ipynb`.


## Contact
In case of questions or comments feel free to raise an issue in this repository.

## License
If you use this work, please cite it using the metadata from `CITATION.cff`.

This repository is released under a BSD-3-Clause license. See `LICENSE` for more details.
