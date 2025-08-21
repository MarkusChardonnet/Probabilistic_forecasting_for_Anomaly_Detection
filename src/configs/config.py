"""
author: Florian Krach
"""

from configs.synthetic.config_AD_dataset import *
from configs.synthetic.config_AD_dataset_anomalies import *
from configs.synthetic.config_AD_model import *
from configs.synthetic.config_AD_experiments import *
from configs.config_microbial_test import *
from configs.config_microbial_dataset import *
from configs.config_microbial_models import *
from configs.config_microbial_AD import *

import numpy as np
import socket

from configs.config_utils import get_parameter_array, makedirs, \
    SendBotMessage, data_path, training_data_path, original_data_path, \
    del_all_flags

if 'ada-' not in socket.gethostname():
    SERVER = False
else:
    SERVER = True

# ==============================================================================
# Global variables
CHAT_ID = "-1002245921777"
ERROR_CHAT_ID = "-4104495192"

SendBotMessage = SendBotMessage
makedirs = makedirs

flagfile = "{}flagfile.tmp".format(data_path)

saved_models_path = '{}saved_models/'.format(data_path)

abx_ts_filename = "ts_vat19_abx_v20240806"
microbial_ft_filename = "ft_vat19_anomaly_v20240806_entero_family.tsv"

hyperparam_default = {
    'drift': 2., 'volatility': 0.3, 'mean': 4, 'poisson_lambda': 3.,
    'speed': 2., 'correlation': 0.5, 'nb_paths': 20000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'scheme': 'euler', 'return_vol': False, 'v0': 1, 'hurst':0.75,
    'FBMmethod':"daviesharte"
}

if __name__ == '__main__':
    pass
