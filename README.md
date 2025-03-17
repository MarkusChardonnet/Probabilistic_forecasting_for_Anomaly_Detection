# Probabilistic Forecasting for Anomaly Detection

In this project, we propose methods for Anomaly Detection in Time Series.
The general approach is Based on Probabilistic Forecasting. The principle is to infer an approximation the conditional distribution of future values knowing the past values of the time series. Then, we exploit the conditional distribution to score individual obsevations of the time series, by computing the p-value. We assume that lower p-values with respect to the approximated conditional distribution are more likely to correspond to anomalies. Finally, scores are weighted and aggregated for different forecasting horizons and neighbouring observations. The aggregated scores are thresholded to raise anomalies.

The probabilistic forecasting is based on the PD-NJODE framework. This framework is described in the paper [Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs](https://arxiv.org/abs/2206.14284) and the code is available at [PD-NJODE](https://github.com/FlorianKrach/PD-NJODE).

## Model inference
### Dependencies
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


### Configurations 
- The configurations can be accessed through the following folder :
    ```bash
    cd src/configs
    ```
- Those are the configurations for generating the synthetic data, training the forecasting and anomaly detection models, plotting and monitoring.

### Generating the data

- Here is an example for the generation of synthetic data from the modified Orstein-Uhlenbeck process : 
    ```bash
    cd src
    python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_3_dict
    ```

- Generating sythetic data with anomalies can be done equivalently by defining other configuration dictionaries :
    ```bash
    cd src
    python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_3_deformation_dict
    ```

- Setting up the dataset from infant microbiome data :
    ```bash
    cd src
    python make_microbial_dataset.py --dataset_config=config_genus
    ```

- Setting up the synthetic infant microbiome dataset :
    ```bash
    cd src
    python data_utils.py --dataset_name=Microbiome_OrnsteinUhlenbeck --dataset_params=config_synthetic_novel_alpha_faith_pd
    ```

### Training the probabilistic forecasting module

Important flags:

- **params**: name of the params list (defined in config.py) to use for parallel run
- **NB_JOBS**: nb of parallel jobs to run with joblib
- **USE_GPU**: whether to use GPU for training
- **NB_CPUS**: nb of CPUs used by each training
- **saved_models_path**: path where the models are saved

- Training with PD-NJODE on the sythetic data :
    ```bash
    cd src
    python run.py --params=param_list_AD_OrnsteinUhlenbeckWithSeason_3 --NB_JOBS=1 --USE_GPU=True --get_overview=overview_dict_AD_OrnsteinUhlenbeckWithSeason_3
    ```

- Training with PD-NJODE on infant microbiome data :
    ```bash
    cd src
    python run.py --params=param_list_microbial_genus --NB_JOBS=1 --USE_GPU=True --GPU_NUM=0
    ```

### Training / Evaluating the Anomaly detection modules on data with ingested anomalies

Important flags:

- **ad_params**: name of the anomaly detection params list (defined in config.py)
- **forecast_saved_models_path**: path where the forecast models are saved
- **forecast_model_ids**: List of forecast model ids to run

- Training and Evaluating the Anomaly Detection Module on synthetic data :
    ```bash
    cd src
    python run.py --forecast_model_ids=[0] --forecast_saved_models_path=../data/saved_models_AD_OrnsteinUhlenbeckWithSeason/ --ad_params=param_dict_AD_modules
    ```

- Evaluating the Anomaly Detection Module on infant microbiome data :
    ```bash
    cd src
    python run.py --forecast_model_ids=[0] --forecast_saved_models_path=../data/saved_models_microbial_genus_base/ --ad_params=param_dict_AD_microbial_genus
    ```



### Training commands (Florian)
#### Generate Dataset:
```shell
python make_microbial_dataset.py --dataset_config=config_otu_sig_highab
python make_microbial_dataset.py --dataset_config=config_genus_sig_highab
```

lower dim datasets, reducing the features with low variance:
```shell
python make_microbial_dataset.py --dataset_config=config_otu_sig_highab_lowvar5
python make_microbial_dataset.py --dataset_config=config_genus_sig_highab_lowvar5
python make_microbial_dataset.py --dataset_config=config_otu_sig_highab_lowvar94q
python make_microbial_dataset.py --dataset_config=config_genus_sig_highab_lowvar94q
```

alpha diversity metric datasets:
```shell
python make_microbial_dataset.py --dataset_config=config_div_alpha_faith_pd_1
python make_microbial_dataset.py --dataset_config=config_div_alpha_faith_pd_2
python make_microbial_dataset.py --dataset_config=config_div_alpha_faith_pd_3
python make_microbial_dataset.py --dataset_config=config_div_alpha_faith_pd_4
python make_microbial_dataset.py --dataset_config=config_div_alpha_faith_pd_5
```

novel alpha diversity metric datasets:
```shell
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

#### Training PD-NJODE:
on OTU dataset:
```shell
python run.py --params=param_list_microbial_otu2 --NB_JOBS=64 --NB_CPUS=1 --SEND=True --USE_GPU=False --first_id=1 --get_overview=overview_dict_microbial_otu2
python run.py --plot_paths=plot_paths_microbial_otu2
```

```shell
python run.py --params=param_list_microbial_otu3 --NB_JOBS=64 --NB_CPUS=1 --SEND=True --USE_GPU=False --first_id=1 --get_overview=overview_dict_microbial_otu3
```

on Genus dataset:
```shell
python run.py --params=param_list_microbial_genus3 --NB_JOBS=64 --NB_CPUS=1 --SEND=True --USE_GPU=False --first_id=1 --get_overview=overview_dict_microbial_genus3
```

lower dim datasets, reducing the features with low variance:
```shell
python run.py --params=param_list_microbial_lowvar --NB_JOBS=64 --NB_CPUS=1 --SEND=True --USE_GPU=False --first_id=1 --get_overview=overview_dict_microbial_lowvar
python run.py --params=param_list_microbial_lowvar1 --NB_JOBS=64 --NB_CPUS=1 --SEND=True --USE_GPU=False --first_id=1 --get_overview=overview_dict_microbial_lowvar1
python run.py --params=param_list_microbial_lowvar2 --NB_JOBS=64 --NB_CPUS=1 --SEND=True --USE_GPU=False --first_id=1 --get_overview=overview_dict_microbial_lowvar2
```

alpha diversity metric datasets:
```shell
python run.py --params=param_list_microbial_alpha_div --NB_JOBS=64 --NB_CPUS=1 --SEND=True --USE_GPU=False --first_id=1 --get_overview=overview_dict_microbial_alpha_div
```

novel alpha diversity metric datasets:
```shell
python run.py --params=param_list_microbial_novel_alpha_div --NB_JOBS=24 --NB_CPUS=1 --SEND=True --USE_GPU=False --first_id=1 --get_overview=overview_dict_microbial_novel_alpha_div

python run.py --params=param_list_microbial_novel_alpha_div2 --NB_JOBS=24 --NB_CPUS=1 --SEND=True --USE_GPU=False --first_id=1 --get_overview=overview_dict_microbial_novel_alpha_div2
python run.py --plot_paths=plot_paths_novel_alpha_div2
```


#### Compute Anomaly Detection Scores:
on OTU dataset:
```shell
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_otu3_ids --ad_params=param_list_AD_microbial_otu --forecast_saved_models_path=AD_microbial_otu3 --compute_scores=True --evaluate_scores=True
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_otu3_ids --ad_params=param_list_AD_microbial_otu1 --forecast_saved_models_path=AD_microbial_otu3 --compute_scores=True --evaluate_scores=True
```

on Genus dataset:
```shell
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_genus3_ids --ad_params=param_list_AD_microbial_genus --forecast_saved_models_path=AD_microbial_genus3 --compute_scores=True --evaluate_scores=True
```

lower dim datasets, reducing the features with low variance:
```shell
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_lowvar_ids --ad_params=param_list_AD_microbial_lowvar --forecast_saved_models_path=AD_microbial_lowvar --compute_scores=True --evaluate_scores=True
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_lowvar1_ids --ad_params=param_list_AD_microbial_lowvar1 --forecast_saved_models_path=AD_microbial_lowvar1 --compute_scores=True --evaluate_scores=True
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_lowvar2_ids --ad_params=param_list_AD_microbial_lowvar2 --forecast_saved_models_path=AD_microbial_lowvar2 --compute_scores=True --evaluate_scores=True

# beta distribution based scoring
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_lowvar2_ids_2 --ad_params=param_list_AD_microbial_lowvar2_2 --forecast_saved_models_path=AD_microbial_lowvar2 --compute_scores=True --evaluate_scores=True
```

alpha diversity metric datasets:
```shell
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_alpha_div_ids --ad_params=param_list_AD_microbial_alpha_div --forecast_saved_models_path=AD_microbial_alpha_div --compute_scores=True --evaluate_scores=True
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_alpha_div_ids_1 --ad_params=param_list_AD_microbial_alpha_div --forecast_saved_models_path=AD_microbial_alpha_div --compute_scores=True --evaluate_scores=True
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_alpha_div_ids_2 --ad_params=param_list_AD_microbial_alpha_div_2 --forecast_saved_models_path=AD_microbial_alpha_div --compute_scores=True --evaluate_scores=True
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_alpha_div_ids_3 --ad_params=param_list_AD_microbial_alpha_div_3 --forecast_saved_models_path=AD_microbial_alpha_div --compute_scores=True --evaluate_scores=True
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_alpha_div_ids_3 --ad_params=param_list_AD_microbial_alpha_div_4 --forecast_saved_models_path=AD_microbial_alpha_div --compute_scores=True --evaluate_scores=True
```

novel alpha diversity metric datasets:
```shell
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_novel_alpha_div_ids --ad_params=param_list_AD_microbial_novel_alpha_div --forecast_saved_models_path=AD_microbial_novel_alpha_div --compute_scores=True --evaluate_scores=True
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_rel_abund_ids --ad_params=param_list_AD_microbial_rel_abund --forecast_saved_models_path=AD_microbial_novel_alpha_div --compute_scores=True --evaluate_scores=True
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_joint_ids --ad_params=param_list_AD_microbial_joint --forecast_saved_models_path=AD_microbial_novel_alpha_div --compute_scores=True --evaluate_scores=True

# reliability evaluation
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_novel_alpha_div_ids_reliability_eval --ad_params=param_list_AD_microbial_novel_alpha_div_reliability_eval --forecast_saved_models_path=AD_microbial_novel_alpha_div --compute_scores=True --evaluate_scores=True
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_novel_alpha_div_ids_reliability_eval_2 --ad_params=param_list_AD_microbial_novel_alpha_div_reliability_eval_2 --forecast_saved_models_path=AD_microbial_novel_alpha_div --compute_scores=True --evaluate_scores=True
```

novel alpha diversity metric datasets with scaling factors:
WARNING: the following things need to be run in the given order to work properly
```shell
# first compute the z-scores and the scaling factors
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_novel_alpha_div_ids2 --ad_params=param_list_AD_microbial_novel_alpha_div2_scaling_factors --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_novel_alpha_div_ids2 --ad_params=param_list_AD_microbial_novel_alpha_div2_scaling_factors2 --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=False --evaluate_scores=False --compute_zscore_scaling_factors=True

# then compute the AD scores using the scaling factors
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2 --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False

# and do a preliminary evaluation of them (only works if AD scores have been computed for only_jump_before_abx_exposure=1,2,3 before)
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_ev --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=False --evaluate_scores=True --compute_zscore_scaling_factors=False

# then compute scores for reliability evaluation using the scaling factors
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_reliability_eval --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False

# as reference, compute scores without scaling factors
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_nsf --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False
python Microbial_AD_eval.py --forecast_model_ids=AD_microbial_novel_alpha_div_ids2_1 --ad_params=param_list_AD_microbial_novel_alpha_div2_reliability_eval_nsf --forecast_saved_models_path=AD_microbial_novel_alpha_div2 --compute_scores=True --evaluate_scores=False --compute_zscore_scaling_factors=False
```


## Score evaluation
Once you trained a microbiome-based model, you can evaluate the resulting scores in the notebook `results/evaluate_scores.ipynb`. To run the notebook, create and activate this conda environment:
````
conda create --name score_eval numpy pandas matplotlib seaborn scipy ipython ipykernel -y
conda activate score_eval
pip install -e .
````
With the same conda environment, a score time horizon reliability analysis can be performed that evaluates for how long the predicted scores remain constant on the no antibiotics scores (notebook `results/analyse_t_horizon_reliability.ipynb`) and the predictive performance of the inferred anomaly scores can be compared to baseline predictions in notebook `results/evaluate_predictions.ipynb`. Also, resulting anomaly scores can be compared to matched alpha diversity values in notebook `evaluate_matched_alpha.ipynb` and individual score trajectories can be explored in `evaluate_indiv_score_increase.ipynb`.

# TODOs and Possible Improvements
- [x] different AD scoring method: using coordinate wise p-values based on beta distribution 
- [x] weighting of coordinates in the loss s.t. they have approx. same sizes
- [x] reducing learning rate during training as @Jakob_Heiss suggested
- [ ] using a different projection approach for NJODE
- [x] or no projection at all
- [x] model the alpha diversity metric -> this leads to best results

- [x] implement scaling of coords
- [x] implement t dist for scoring and plotting
- [x] implement coordinate wise scoring
- [x] implement plotting of selected (instead of all) dists
- [x] run

**14.03.2025**:
- [x] fix the bug in the real dataset generation that caused the dynamic features not to be used
- [] rerun model training with fixed bug in dataset generation that caused dynamic features not to be used
- [] fix the synthetic dataset generation to have the same number of dynamic features as the real dataset
- [] fix the synthetic dataset generation to have same type of output as the real dataset
- [] train model on synthetic dataset (of same size as real one) and evaluate it on a very large synthetic dataset to have better statistics
- [] maybe: train model on larger synthetic dataset and see whether it performs better on the same evaluation dataset as above
- [] maybe (as followup work?): train model first on large synthetic dataset and then on real dataset to see whether it can learn the real dataset better