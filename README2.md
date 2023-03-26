# Probabilistic Forecasting for Anomaly Detection

In this project, we propose methods for Anomaly Detection in Time Series.
The general approach is Based on Probabilistic Forecasting. The principle is to infer the conditional distribution of future values knowing the past values of the time series.
The idea is to exploit the conditional distribution to raise Anomalies for unlikely observations, i.e when observations fall in tails of the conditional distribution, or regions with small likelihood.
Hereby, we build models that are able to learn the conditional distribution from data that is considered normal, i.e without anomalies.

The first model is a Gaussian Invariant Linear State Space model. The state is present to carry information from past observations. 

### Dependencies
- Python 3.10 and Conda
- Create the environment and install dependencies:
  ```bash
    conda env create -f environment.yml
    ```
- Activate environment:
    ```bash
    conda activate adts
    ```

### Experiments
- To run the first experiment:
    ```bash
    python experiments.py
    ```
    In this experiment, we generate data from an ARMA process, which can be written in the form of a Gaussian Invariant State Space model. We learn an model by optimizing the likelihood of the data. Then, we compare the predictive distribution of the learned model with the one of the true model.