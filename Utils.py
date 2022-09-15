import numpy as np

def get_dynamic_threshold(es):
    candidate_thrs = []
    max = -np.inf
    argmax = np.inf
    for z in np.linspace(0.5, 10, 20):
        candidate_thrs.append(np.mean(es) + z*np.std(es))
    mu_e = np.mean(es)
    sigma_e = np.std(es)
    for thrs in candidate_thrs:
        fltr = es[es > thrs]
        if not len(fltr):
            continue
        d_mu = mu_e - np.mean(fltr)
        d_sigma = sigma_e - np.std(es)
        test = (d_mu/mu_e + d_sigma/sigma_e)/(len(fltr) + len(es)**2)
        if test > max:
            max = test
            argmax = thrs
    return argmax
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))
def get_e(ys, yhats, err_fnc):
    e = []
    for y, yhat in zip(ys, yhats):
        e.append(err_fnc(y, yhat))
    return np.array(e)
def classify_pl(y, yhat, window_length):
    assert(y.shape[0] == yhat.shape[0])
    err_fnc = rmse
    e = get_e(y, yhat, err_fnc)
    labels = np.zeros(y.shape[0])

    for i in range(0, e.shape[0], window_length):
        es = e[i:i+ window_length]
        thrs = get_dynamic_threshold(es)
        labels[i:i + window_length] = es > thrs
    return labels

