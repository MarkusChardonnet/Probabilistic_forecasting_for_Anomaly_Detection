# Probabilistic Forecasting for Anomaly Detection

In this project, we propose methods for Anomaly Detection in Time Series.
The general approach is Based on Probabilistic Forecasting. The principle is to infer an approximation the conditional distribution of future values knowing the past values of the time series. Then, we exploit the conditional distribution to score individual obsevations of the time series, by computing the p-value. We assume that lower p-values with respect to the approximated conditional distribution are more likely to correspond to anomalies. Finally, scores are weighted and aggregated for different forecasting horizons and neighbouring observations. The aggregated scores are thresholded to raise anomalies.

The probabilistic forecasting is based on the PD-NJODE framework. This framework is described in the paper [Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs](https://arxiv.org/abs/2206.14284) and the code is available at [PD-NJODE](https://github.com/FlorianKrach/PD-NJODE).

### Dependencies
- Python 3.7 and Conda
- Create the environment and install dependencies:
  ```bash
    conda env create -f environment.yml
    ```
- Activate environment:
    ```bash
    conda activate pbforecast
    ```

### Configurations 
The configurations for both the synthetic data and the training can be accessed through the following folder :
    ```bash
    cd PD-NJODE/NJODE/configs
    ```

### Generating the data
Here is an example for the generation of the modified Orstein-Uhlenbeck process : 
    ```bash
    cd PD-NJODE/NJODE
    python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_3_dict
    ```

Generating sythetic data with anomalies can be done equivalently by defining other configuration dictionaries :
    ```bash
    cd PD-NJODE/NJODE
    python data_utils.py --dataset_name=AD_OrnsteinUhlenbeckWithSeason --dataset_params=AD_OrnsteinUhlenbeckWithSeason_3_deformation_dict
    ```

### Training the probabilistic forecasting module
Training with PD-NJODE on the sythetic data :
    ```bash
    cd PD-NJODE/NJODE
    python run.py --params=param_list_AD_OrnsteinUhlenbeckWithSeason_3 --NB_JOBS=4 --SEND=0 --USE_GPU=True --get_overview=overview_dict_AD_OrnsteinUhlenbeckWithSeason_3
    ```

### Training the Anomaly detection modules on data with ingested anomalies

### Evaluating the Anomaly Detection pipeline