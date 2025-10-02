# Probabilistic Forecasting for Anomaly Detection

In this project, we propose a new framework for anomaly detection in irregularly sampled multi-variate time-series and apply it to synthetic data and infant gut microbiome trajectories. The method is based on neural jump ordinary differential equations (NJODEs) and infers conditional mean and variance trajectories in a fully path dependent way to compute anomaly scores. Read more about this study in our preprint [here](https://arxiv.org/abs/2510.00087):

Adamov, A., Chardonnet, M., Krach, F., Heiss, J., Teichmann, J., Bokulich, N. A. Revealing the Temporal Dynamics of Antibiotic Anomalies in the Infant Gut Microbiome with Neural Jump ODEs. 2025. doi: 
https://doi.org/10.48550/arXiv.2510.00087


## Dependencies
- Python 3.7 and Conda
- Create environment in python 3.7:
  ```bash
    conda create -n pbforecast python=3.7
    ```
- Activate environment:
    ```bash
    conda activate pbforecast
    ```
- Install dependencies:
  ```bash
    pip install numpy==1.18.5
    pip install -r requirements.txt
    ```
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
    python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_3_dict --seed=0
    python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_eval --seed=1
    ```

  - Generation of synthetic data with anomalies:
      ```bash
      cd src
      python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_diffusion_dict --seed=0
      python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_diffusion_dict_eval --seed=1
      python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_noise_dict --seed=0
      python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_noise_dict_eval --seed=1
      python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_cutoff_dict --seed=0
      python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_cutoff_dict_eval --seed=1
      python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_spike_dict --seed=0
      python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_spike_dict_eval --seed=1
      ```
  - Generation of plots of base and anomaly datasets:
    ```bash
    python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_plot --seed=0 --plot_paths=[0]
    python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_diffusion_dict_plot --seed=0 --plot_paths=[0]
    python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_noise_dict_plot --seed=0 --plot_paths=[0] 
    python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_cutoff_dict_plot --seed=0 --plot_paths=[0]
    python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_spike_dict_plot --seed=0 --plot_paths=[0]
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
    python run.py --params=param_list_AD_OrnsteinUhlenbeckWithSeason_3 --NB_JOBS=1 --NB_CPUS=1 --USE_GPU=False --get_overview=overview_dict_AD_OrnsteinUhlenbeckWithSeason    
    ```


## Training / Evaluating the Anomaly detection modules on data with ingested anomalies

Important flags:

- **ad_params**: name of the anomaly detection params list (defined in config.py)
- **forecast_saved_models_path**: path where the forecast models are saved
- **forecast_model_ids**: List of forecast model ids to run

- Training and Evaluating the Anomaly Detection Module on synthetic data :
    ```bash
    cd src
    python evaluate_AD.py --forecast_model_ids=[1] --forecast_saved_models_path=../data/saved_models_AD_OrnsteinUhlenbeckWithSeason/ --ad_params=param_dict_AD_modules
    # plot from the last (or best) AD model
    python evaluate_AD.py --plot_only=True --quant_eval_in_plot_only=True --which_AD_model_load="last" --forecast_model_ids=[1] --forecast_saved_models_path=../data/saved_models_AD_OrnsteinUhlenbeckWithSeason/ --ad_params=param_dict_AD_modules
    ```

---
# Anomaly Detection in Microbiome Data
## Generate Dataset:

novel alpha diversity metric datasets:
```shell
# the following is the dataset which is used for the final evaluation
python make_microbial_dataset.py --dataset_config=config_novel_alpha_faith_pd
python make_microbial_dataset.py --dataset_config=config_novel_alpha_faith_pd_w_geo

python make_microbial_dataset.py --dataset_config=config_entero_family
python make_microbial_dataset.py --dataset_config=config_entero_family_w_geo
python make_microbial_dataset.py --dataset_config=config_entero_genus
python make_microbial_dataset.py --dataset_config=config_entero_genus_w_geo

python make_microbial_dataset.py --dataset_config=config_novel_alpha_faith_pd_entero_family
python make_microbial_dataset.py --dataset_config=config_novel_alpha_faith_pd_entero_genus
python make_microbial_dataset.py --dataset_config=config_novel_alpha_faith_pd_entero_genus_scaled
```


## Training PD-NJODE:
novel alpha diversity metric datasets:
```shell
python run.py --params=param_list_microbial_novel_alpha_div --NB_JOBS=24 --NB_CPUS=1 --SEND=True --USE_GPU=False --first_id=1 --get_overview=overview_dict_microbial_novel_alpha_div

# the following is the model training which is used for the final evaluation
python run.py --params=param_list_microbial_novel_alpha_div2 --NB_JOBS=24 --NB_CPUS=1 --SEND=True --USE_GPU=False --first_id=1 --get_overview=overview_dict_microbial_novel_alpha_div2
python run.py --plot_paths=plot_paths_novel_alpha_div2
```

Based on the training results, the model ids 55,56,57 (using RNN, no signature and increasing probability to only use dynamic features as inputs; with the 3 different validation horizons) are selected. Computing the outputs on the validation set shows that only 57 never predicts a cond variance <0. 
Moreover, this model produces increasing confidence interval sizes and it has the best eval loss (i.e. train loss on val set) of the 3 models. Therefore, this model is selected for further evaluation.


## Compute Anomaly Detection Scores:

novel alpha diversity metric datasets with scaling factors:
WARNING: the following things need to be run in the given order to work properly
```shell
# first compute the z-scores and the scaling factors
python Microbial_AD_eval.py --NB_CPUS=1 --forecast_model_ids=AD_microbial_novel_alpha_div_ids2 --ad_params=param_list_AD_microbial_novel_alpha_div2_scaling_factors --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False
python Microbial_AD_eval.py --NB_CPUS=1 --forecast_model_ids=AD_microbial_novel_alpha_div_ids2 --ad_params=param_list_AD_microbial_novel_alpha_div2_scaling_factors2 --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=False --evaluate_scores=False --compute_zscore_scaling_factors=True

# then compute the AD scores using the scaling factors
python Microbial_AD_eval.py --NB_CPUS=1 --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2 --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False

# and do a preliminary evaluation of them (only works if AD scores have been computed for only_jump_before_abx_exposure=1,2,3 before)
python Microbial_AD_eval.py --NB_CPUS=1 --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_ev --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=False --evaluate_scores=True --compute_zscore_scaling_factors=False

# then compute scores for reliability evaluation using the scaling factors
python Microbial_AD_eval.py --NB_CPUS=1 --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_reliability_eval --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False

# as reference, compute scores without scaling factors
python Microbial_AD_eval.py --NB_CPUS=1 --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_nsf --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False
python Microbial_AD_eval.py --NB_CPUS=1 --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_reliability_eval_nsf --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False
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
