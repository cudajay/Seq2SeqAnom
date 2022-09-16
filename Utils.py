import numpy as np
import pandas as pd
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
        #d_mu = mu_e - np.mean(fltr)
        #d_sigma = sigma_e - np.std(es)
        #test = (d_mu/mu_e + d_sigma/sigma_e)/(len(fltr) + len(es)**2)
        test = mu_e + thrs*sigma_e
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
    plabels = np.zeros(y.shape[0])

    for i in range(0, e.shape[0], window_length):
        es = e[i:i+ window_length]
        thrs = get_dynamic_threshold(es)
        plabels[i:i + window_length] = es > thrs
    return plabels

def str2ary(str_):
    x = str_.replace("]","").replace("[","")
    x = x.split(",")
    assert(not len(x)%2)
    lst = []
    for i in range(0, len(x), 2):
        tmp = tuple([int(x[i]), int(x[i+1])])
        lst.append(tmp)
    return lst

def make_discrete_lbls(lbl_path):
    lbls = pd.read_csv(lbl_path)
    labels_dict = {}
    for rw in lbls.iterrows():
        labels_dict[rw[1]['chan_id']] = str2ary(rw[1]['anomaly_sequences'])
    return labels_dict

def vectorize_labels(lst_of_range_tuples, nrows):
    ret = np.zeros(nrows)
    for t in lst_of_range_tuples:
        ret[t[0]:t[1]] = 1
    return ret