# Probabilistic Forecasting for Anomaly Detection

In this project, we propose methods for Anomaly Detection in Time Series.
The general approach is Based on Probabilistic Forecasting. The principle is to infer an approximation the conditional distribution of future values knowing the past values of the time series. Then, we exploit the conditional distribution to score individual obsevations of the time series, by computing the p-value. We assume that lower p-values with respect to the approximated conditional distribution are more likely to correspond to anomalies. Finally, scores are weighted and aggregated for different forecasting horizons and neighbouring observations. The aggregated scores are thresholded to raise anomalies.

The probabilistic forecasting is based on the PD-NJODE framework. This framework is described in the paper [Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs](https://arxiv.org/abs/2206.14284) and the code is available at [PD-NJODE](https://github.com/FlorianKrach/PD-NJODE).

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

- **ad_params**: name of the anomaly deteaction params list (defined in config.py)
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



## Training commands (Florian)
#### Generate Dataset:
```shell
python make_microbial_dataset.py --dataset_config=config_otu_sig_highab
```

#### Training PD-NJODE:
```shell
python run.py --params=param_list_microbial_otu2 --NB_JOBS=64 --NB_CPUS=1 --SEND=True --USE_GPU=False --first_id=1
```