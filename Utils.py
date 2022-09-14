import numpy as np

def get_dynamic_threshold(e):
    candidate_thrs = []
    max = -np.inf
    argmax = np.inf
    for z in np.linspace(0.5, 10, 20):
        candidate_thrs.append(np.mean(e) + z*np.std(e))
    mu_e = np.mean(e)
    sigma_e = np.std(e)
    for thrs in candidate_thrs:
        fltr = e[e > thrs]
        if  not len(fltr):
            continue
        d_mu = mu_e - np.mean(fltr)
        d_sigma = sigma_e - np.std(e)
        test = (d_mu/mu_e + d_sigma/sigma_e)/(len(fltr) + len(e)**2)
        if test > max:
            max = test
            argmax = thrs
    return argmax

def classifier_pipe(y, yhat):
    e = np.abs(y - yhat)  
    thrs = get_dynamic_threshold