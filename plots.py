import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as ss

def visualize_time_series(time_series, path, indices = None):
    if indices is None:
        indices = np.arange(time_series.shape[1])

    for i in range(time_series.shape[0]):
        plt.plot(indices, time_series[i])
    plt.savefig(path)
    plt.close()

def visualize_forecasting(ts, estims, path, steps, labels = None, washout = 50, indices = None, var_factor = 1.96):
    if indices is None:
        indices = np.arange(ts.shape[1])

    colors = np.random.random(size=(len(estims),3))
    colors = list(map(tuple, colors))
    colors[0] = 'r'
    colors[1] = 'g'

    for i in range(ts.shape[0]):
        plt.plot(indices, ts[i], 'b', label="true data")

        for j, e in enumerate(estims):
            estim, estim_var = e
            label = None
            if labels is not None:
                label = labels[j] + " estimation"
            es = np.concatenate((np.empty(shape=washout), np.array(estim[i])),axis=0)
            masked_estim = np.ma.masked_where((indices<washout), es)
            plt.plot(indices, masked_estim, colors[j], label=label)
            if estim_var is not None:
                if labels is not None:
                    label = labels[j] + " covariance error"
                vs1 = np.concatenate((np.empty(shape=washout), np.array(estim[i]) - var_factor * np.sqrt(np.array(estim_var[i]))),axis=0)
                vs2 = np.concatenate((np.empty(shape=washout), np.array(estim[i]) + var_factor * np.sqrt(np.array(estim_var[i]))),axis=0)
                masked_vs1 = np.ma.masked_where((indices<washout), vs1)
                masked_vs2 = np.ma.masked_where((indices<washout), vs2)
                plt.fill_between(indices, masked_vs1, masked_vs2, color=colors[j], alpha=.3, label=label)

        plt.title("Prediction, {} steps ahead".format(steps))
        plt.legend(loc="upper right")
        plt.savefig(path + "_sample_{}.png".format(i+1))
        plt.close()

def visualize_innov_distribution(innov_samples, innov_variances, path, labels = None, washout = 0):
    b = innov_samples[0].shape[0]

    Pts = [[] for j in range(len(innov_samples))]
    for i in range(b):
        for s in range(len(innov_samples)):
            samples = innov_samples[s][i,washout:]
            stds = np.sqrt(innov_variances[s][i,washout:])
            pts = samples / stds
            Pts[s] = Pts[s] + list(pts.reshape(-1))
    Pts = np.array(Pts)
    print(Pts.shape)
    fig, axs = plt.subplots(1, len(innov_samples), sharey=True, tight_layout=True)
    for s in range(len(innov_samples)):
        bins = 100
        x = np.linspace(-5,5,bins)
        rv = ss.norm()
        axs[s].hist(Pts[s], range = (-5,5), density=True, bins=bins, label="Innovation samples")
        axs[s].plot(x, rv.pdf(x), label="Innovation Distribution")
        if labels is not None:
            axs[s].set_title(labels[s])
    fig.suptitle("Kalman innovations against theoretical distribution")
    fig.savefig(path)
    plt.close()
    return
