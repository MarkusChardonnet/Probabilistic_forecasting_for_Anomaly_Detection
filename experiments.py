import numpy as np
import torch
import os
from datetime import datetime

from synthetic_data import generate_arma_process, arma_2_ssm_params, generate_stationary_ssm_params
from plots import visualize_time_series, visualize_forecasting, visualize_innov_distribution
from state_space_model import GaussianLinearInvariantSSM

"""
def main():
    length = 200
    nrealizations = 2
    arparams = np.array([.75, -.25])
    maparams = np.array([.65, .35])
    noise_var = 1.
    mean = 0.

    # path = "plots/arma_p{}_q{}_sample{}".format(len(arparams),len(maparams),1)
    input = generate_arma_process(arparams=arparams,maparams=maparams,scale=noise_var,length=length,nts=nrealizations)
    input = torch.tensor(input, dtype=torch.float32).view(nrealizations,-1,1)

    washout = 50
    steps_ahead = 2

    initial_params, d_obs, d_state, d_noise = arma_2_ssm_params(arparams, maparams, noise_var, mean)
    d_obs = 1
    d_noise = 1
    d_state = 5
    true_model = GaussianLinearStationarySSM(d_obs=d_obs, d_state=d_state, d_noise=d_noise, initial_params=initial_params)
    estim_exps, estim_vars = true_model.forecast(input, steps_ahead = steps_ahead, washout = washout, time_horizon = None)
    estim_exps = torch.cat([e.unsqueeze(1) for e in estim_exps],dim=1).view(nrealizations,-1).detach().numpy()
    estim_vars = torch.cat([v.unsqueeze(1) for v in estim_vars],dim=1).view(nrealizations,-1).detach().numpy()

    input = input.view(nrealizations,-1)
    path = "plots/arma_p{}_q{}_estim_ssm".format(len(arparams),len(maparams))
    visualize_forecasting(input, estim_exps, path, washout = washout + steps_ahead - 1, estim_var = estim_vars, indices = None)
"""

def get_data(process_config, length = 1000, nts = 1000):

    if process_config["type"] == 'ARMA':
        params = process_config["params"]
        arparams = np.array(params["arparams"])
        maparams = np.array(params["maparams"])
        noise_var = params["noise_var"]
        mean = params["mean"]

        data = generate_arma_process(arparams=arparams,maparams=maparams,scale=noise_var,length=length,nts=nts)
        data = torch.tensor(data, dtype=torch.float32).view(nts,-1,1)

        initial_params, d_obs, d_state, d_noise = arma_2_ssm_params(arparams, maparams, noise_var, mean)
        ssm_dims = (d_obs, d_state, d_noise)
        print("Number of samples : {}".format(nts))

    print("Loaded data")

    return data, initial_params, ssm_dims

def train(data, initial_params, ssm_dims, epochs = 50, val_loop = 10, train_val = (0.8,0.2), lr = 0.001, batch_size = 32):
    d_obs, d_state, d_noise = ssm_dims
    true_model = GaussianLinearInvariantSSM(d_obs=d_obs, d_state=d_state, d_noise=d_noise, initial_params=initial_params)
    model = GaussianLinearInvariantSSM(d_obs=d_obs, d_state=d_state, d_noise=d_noise)
    params2freeze = ['a']
    model.freeze_params(params2freeze)

    train_dataloader_params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}
    val_dataloader_params = {'batch_size': batch_size,
          'shuffle': False,
          'num_workers': 6}

    nsamples = data.size(0)
    train_set, val_set = torch.utils.data.random_split(data, [int(nsamples * train_val[0]), int(nsamples * train_val[1])])
    training_loader = torch.utils.data.DataLoader(train_set, **train_dataloader_params)
    validation_loader = torch.utils.data.DataLoader(val_set, **val_dataloader_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Begin training..")

    for e in range(epochs):

        print("Epoch {}".format(e+1))

        best_nll = float('inf')

        running_nll = 0
        true_running_nll = 0

        for i, data_i in enumerate(training_loader):
            print("Batch {}".format(i+1))
            optimizer.zero_grad()
            nll = model.negloglikelihood(data_i).mean()
            true_nll = true_model.negloglikelihood(data_i).mean()
            nll.backward()
            optimizer.step()

            running_nll += nll
            true_running_nll += true_nll
            
            if i % val_loop == val_loop - 1:
                print("Batch {} to {} mean log-likelihood with CURRENT model : {}".format(i-19,i+1,-running_nll / val_loop))
                print("And with TRUE model : {}".format(-true_running_nll / val_loop))
                running_nll = 0
                true_running_nll = 0

        running_nll = 0
        n = 0
        for j, data_j in enumerate(validation_loader):
            nll = model.negloglikelihood(data_j).mean()
            n += 1
            running_nll += nll
        running_nll /= n
        print("Mean log-likelihood is of the model on validation set is {}".format(-running_nll))
        if running_nll < best_nll:
            print("Saving model ..")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = 'models/model_date_{}'.format(timestamp)
            torch.save(model.state_dict(), model_path)

    print("Training finished ! ")

    true_running_nll = 0
    n = 0
    for j, data_j in enumerate(validation_loader):
        true_nll = true_model.negloglikelihood(data_j).mean()
        n += 1
        true_running_nll += true_nll
    true_running_nll /= n
    print("Mean log-likelihood is of the BEST model on validation set is {}".format(-best_nll))
    print("Mean log-likelihood is of the TRUE model on validation set is {}".format(-true_running_nll))


def test(data, process_config, initial_params, ssm_dims):

    nts = data.size(0)
    d_obs, d_state, d_noise = ssm_dims

    path = 'models/model_date_20230207_090946'
    model = GaussianLinearInvariantSSM(d_obs=d_obs, d_state=d_state, d_noise=d_noise)
    model.load_state_dict(torch.load(path))
    true_model = GaussianLinearInvariantSSM(d_obs=d_obs, d_state=d_state, d_noise=d_noise, initial_params=initial_params)
    
    washout = 50
    steps_ahead = 2
    
    estim_exps, estim_vars = true_model.forecast(data, steps_ahead = steps_ahead, washout = washout, time_horizon = None)
    estim_exps = torch.cat([e.unsqueeze(1) for e in estim_exps],dim=1).view(nts,-1).detach().numpy()
    estim_vars = torch.cat([v.unsqueeze(1) for v in estim_vars],dim=1).view(nts,-1).detach().numpy()
    estims = [(estim_exps,estim_vars)]

    estim_exps, estim_vars = model.forecast(data, steps_ahead = steps_ahead, washout = washout, time_horizon = None)
    estim_exps = torch.cat([e.unsqueeze(1) for e in estim_exps],dim=1).view(nts,-1).detach().numpy()
    estim_vars = torch.cat([v.unsqueeze(1) for v in estim_vars],dim=1).view(nts,-1).detach().numpy()
    estims.append((estim_exps,estim_vars))

    innovs, innov_vars, _, _ = model.filter_sequence(data, data.size(1))
    true_innovs, true_innov_vars, _, _ = true_model.filter_sequence(data, data.size(1))
    innovs = torch.cat([e.unsqueeze(1) for e in innovs],dim=1).view(nts,-1).detach().numpy()
    innov_vars = torch.cat([e.unsqueeze(1) for e in innov_vars],dim=1).view(nts,-1).detach().numpy()
    true_innovs = torch.cat([e.unsqueeze(1) for e in true_innovs],dim=1).view(nts,-1).detach().numpy()
    true_innov_vars = torch.cat([e.unsqueeze(1) for e in true_innov_vars],dim=1).view(nts,-1).detach().numpy()

    process_type = process_config["type"]
    p = process_config["params"]["p"]
    q = process_config["params"]["q"]
    process_name = process_type + '(' + str(p) + ',' + str(q) + ')'
    base_path = "plots" + os.sep + process_name
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    labels = ["Learned model", "True model"]

    innov_samples = [innovs, true_innovs]
    innov_variances = [innov_vars, true_innov_vars]
    path = base_path + os.sep + "ssm_innovation_distribution"
    print(path)
    visualize_innov_distribution(innov_samples, innov_variances, path, labels, washout)

    data = data.view(nts,-1)
    path = base_path + os.sep + "ssm_forecasting_{}_steps_ahead".format(steps_ahead)
    visualize_forecasting(data, estims, path, steps_ahead, labels, washout = washout + steps_ahead - 1, indices = None)


if __name__== "__main__" :

    process_config = {
        "type": "ARMA",
        "params": {
            "arparams": [.75, -.25],
            "maparams": [.65, .35],
            "noise_var": 1.,
            "mean": 0.,
            "p": 2,
            "q": 2,
        }
    }
    # data, initial_params, dims = get_data(length=500, nts=10000)
    # train(data, initial_params, dims, epochs=2, train_val=(0.95,0.05))
    # data, initial_params, ssm_dims = get_data(process_config, nts=10, length=200)
    # test(data, process_config, initial_params, ssm_dims)


    d_obs = 1
    d_state = 5
    d_noise = 1
    nsamples = 1
    length = 200
    initial_params = generate_stationary_ssm_params(d_obs=d_obs,d_noise=d_noise,d_state=d_state)
    model = GaussianLinearInvariantSSM(d_obs=d_obs, d_state=d_state, d_noise=d_noise, initial_params=initial_params)
    base_path = 'PD-NJODE/NJODE/configs/models'
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    model.load_model(base_path + os.sep + 'AD_GLISSM_1.pt')
    data = model.generate_data(nsamples=nsamples, ts_length=length)
    data = data.squeeze(dim=2).detach().numpy()
    visualize_time_series(time_series=data, path='test1.png')

    path = 'PD-NJODE/data/training_data/AD_TSAGen-9/'
    with open('{}data.npy'.format(path), 'rb') as f:
        stock_paths = np.load(f)
        observed_dates = np.load(f)
        nb_obs = np.load(f)
    print(stock_paths.shape)
    print(observed_dates.shape)
    print(nb_obs.shape)
    data = stock_paths[0]
    visualize_time_series(time_series=data, path='test2.png')
    # model.save_model(base_path + os.sep + 'AD_GLISSM_1.pt')

