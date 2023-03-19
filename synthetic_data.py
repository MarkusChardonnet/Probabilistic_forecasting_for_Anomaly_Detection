import numpy as np
import statsmodels.api as sm
import scipy.linalg as la
import torch

# def generate_arma_params(p,q):


def generate_arma_process(arparams, maparams, length, scale=1., nts=1):

    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    ARMA = sm.tsa.ArmaProcess(ar=ar, ma=ma)
    X = np.empty([nts,length])
    for i in range(nts):
        x = ARMA.generate_sample(nsample=length, scale=scale)
        X[i] = x

    return X

def arma_2_ssm_params(arparams, maparams, noise_var = 1., mean = 0.):
    p = len(arparams)
    q = len(maparams)
    r = max(p,q+1)

    arp = torch.zeros(r)
    arp[:p] = torch.tensor(arparams)
    map = torch.zeros(r)
    map[1:q+1] = torch.tensor(maparams)

    A = torch.zeros(r,r)
    A[1:r,0:r-1] = torch.eye(r-1)
    A[0] = arp
    B = torch.zeros(1,r)
    B[0] = map
    B[0,0] = 1.
    W = torch.zeros(r,1)
    W[0,0] = 1
    U = torch.zeros(1,1)
    U[0,0] = noise_var
    b = torch.zeros(1)
    b[0] = mean

    initial_params = {
        'a': torch.zeros(r),
        'b': b,
        'A': A,
        'B': B,
        'U': U,
        'V': torch.zeros(1,1),
        'W': W,
        'a0': torch.zeros(r),
        'A0': torch.eye(r, r)
    }

    d_obs = 1
    d_state = r
    d_noise = 1

    return initial_params, d_obs, d_state, d_noise

def generate_stationary_ssm_params(d_obs, 
                                   d_state, 
                                   d_noise,
                                   max_range_dict = {}):

    # according to Piet de Jong et. al.
    # One sufficient condition for a time-invariant SSM to be stationary (both mean and covariance stationary)
    # is if A (state transition matrix) has only stationary roots (i.e eigenvalues)
    # and if equations from starting parameters are satisfied
    # this means each of them has absolute value less than 1
    # we generate a random matrix A by first generating its eigenvalues
    # and then conjugating it with an orthonormal matrix

    # Random A
    eig_val_A = np.random.rand(d_state)  # random values in interval [0,1)
    neg_eig_val_A = np.random.choice(a=[False, True], size=d_state)
    eig_val_A[neg_eig_val_A] = -eig_val_A[neg_eig_val_A] # a random subset should be negative so to have values in (-1,1) interval
    Q, _ = la.qr(np.random.rand(d_state, d_state)) # generate random orthonormal matrix
    A = Q.T @ np.diag(eig_val_A) @ Q

    if "U" in max_range_dict.keys():
        max_U_range = max_range_dict["U"]
    else:
        max_U_range = 1   

    if "V" in max_range_dict.keys():
        max_V_range = max_range_dict["V"]
    else:
        max_V_range = 1  

    if "W" in max_range_dict.keys():
        max_W_range = max_range_dict["W"]
    else:
        max_W_range = 1  

    if "B" in max_range_dict.keys():
        max_B_range = max_range_dict["B"]
    else:
        max_B_range = 1  

    if "a" in max_range_dict.keys():
        max_a_range = max_range_dict["a"]
    else:
        max_a_range = 1 

    if "b" in max_range_dict.keys():
        max_b_range = max_range_dict["b"]
    else:
        max_b_range = 1                
    

    # Random U, V, W, B, a, b
    V = np.random.uniform(low=-max_V_range, high=max_V_range, size=(d_obs, d_obs))
    U = np.random.uniform(low=-max_U_range, high=max_U_range, size=(d_noise, d_noise))
    W = np.random.uniform(low=-max_W_range, high=max_W_range, size=(d_state, d_noise))
    B = np.random.uniform(low=-max_B_range, high=max_B_range, size=(d_obs, d_state))
    a = np.random.uniform(low=-max_a_range, high=max_a_range, size=(d_state))
    b = np.random.uniform(low=-max_b_range, high=max_b_range, size=(d_obs))

    # Solve for A0, a0
    A0 = np.random.uniform(low=-max_W_range, high=max_W_range, size=(d_state, d_state))
    a0 = np.random.uniform(low=-max_W_range, high=max_W_range, size=(d_state))

    initial_params = {
        'a': torch.tensor(a, dtype=torch.float32),
        'b': torch.tensor(b, dtype=torch.float32),
        'A': torch.tensor(A, dtype=torch.float32),
        'B': torch.tensor(B, dtype=torch.float32),
        'U': torch.tensor(U, dtype=torch.float32),
        'V': torch.tensor(V, dtype=torch.float32),
        'W': torch.tensor(W, dtype=torch.float32),
        'a0': torch.tensor(a0, dtype=torch.float32),
        'A0': torch.tensor(A0, dtype=torch.float32)
    }

    return initial_params