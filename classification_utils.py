import mat73
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from centers_utils import * 

# definition from https://ieeexplore.ieee.org/document/8013808
def riemannian_distance(X,Y):
    X_invsqrtm = invsqrtm(X)
    if X.ndim > 2 or Y.ndim > 2:
        return np.sum(np.log(np.linalg.eigvalsh(X_invsqrtm @ Y @ X_invsqrtm))**2, axis=1)
    else:
        return np.sum(np.log(np.linalg.eigvalsh(X_invsqrtm @ Y @ X_invsqrtm))**2)

# based on riemannian distance
# input is the centers of the two classes and the data to test
# class 0 is nontarget, class 1 is target - shape should be n_trials x n_cov^2 
# definition from https://ieeexplore.ieee.org/document/8013808
def mdm_riemann_classifier(center_class0, center_class1, x_test, y_test): 
    n_trials, n_cov, _ = x_test.shape

    pred = []
    for trial_idx in range(n_trials):
        dist_from_0 = riemannian_distance(center_class0, x_test[trial_idx])
        dist_from_1 = riemannian_distance(center_class1, x_test[trial_idx])
    
        if dist_from_0 <= dist_from_1:
            pred.append(0)
        else:
            pred.append(1)
    
    return np.asarray(pred)

def get_confusion_mat(y_pred, y_true, n_nontarget, n_target, if_plot=False):
    true_neg = np.sum((y_pred[:n_nontarget] == 0) & (y_true[:n_nontarget] == 0))
    false_neg = np.sum((y_pred[n_nontarget:] == 0) & (y_true[n_nontarget:] == 1))
    false_pos = np.sum((y_pred[:n_nontarget] == 1) & (y_true[:n_nontarget] == 0))
    true_pos = np.sum((y_pred[n_nontarget:] == 1) & (y_true[n_nontarget:] == 1))

    cf_mat = np.asarray([[true_neg, false_pos], [false_neg, true_pos]])

    if if_plot == True:
        disp = ConfusionMatrixDisplay(cf_mat, display_labels=["Non-target", "Target"])
        disp.plot()
        plt.show()
    
    return cf_mat

def get_stats(cf_mat):
    n_trials = np.sum(cf_mat)
    n_nontarget, n_target = np.sum(cf_mat, axis=1)

    acc = (cf_mat[0,0] + cf_mat[1,1]) / n_trials    # true neg + true pos 
    sens = (cf_mat[1,1]) / n_target                 # tested pos given actually pos
    spec = (cf_mat[0,0]) / n_nontarget              # tested neg given actually neg

    return [acc, sens, spec]

# below is helper function
def transform_mat(mat, ref):
    ref_invsqrt = invsqrtm(ref)
    return ref_invsqrt @ mat @ ref_invsqrt

# from: https://ieeexplore.ieee.org/document/8013808
# input is the data from both classes from two sessions/players/tasks/etc 
def transfer_learn(nontarget0, target0, nontarget1, target1, cent_type):
    if cent_type == 'riemann_mean':
        ref_0 = riemannian_mean(nontarget0)
        ref_1 = riemannian_mean(nontarget1)
    elif cent_type == 'euclid_mean':
        ref_0 = euclidean_mean(nontarget0)
        ref_1 = euclidean_mean(nontarget1)
    elif cent_type == 'mat_med':
        ref_0 = matrix_median(nontarget0)
        ref_1 = matrix_median(nontarget1)
    elif cent_type == 'riemann_med':
        ref_0 = riemannian_median(nontarget0)
        ref_1 = riemannian_median(nontarget1)
    elif cent_type == 'huber':
        ref_0 = huber_centroid(nontarget0)
        ref_1 = huber_centroid(nontarget1)
    else:
        raise ValueError("No such center type. Options are: riemann_mean, euclid_mean, mat_med, riemann_med, and huber")

    new_nontarget0 = transform_mat(nontarget0, ref_0)
    new_target0 = transform_mat(target0, ref_0)
    new_nontarget1 = transform_mat(nontarget1, ref_1)
    new_target1 = transform_mat(target1, ref_1)

    return new_nontarget0, new_target0, new_nontarget1, new_target1

def get_class_results(train_nontarget_cov, train_target_cov, test_nontarget_cov, test_target_cov, tl_bool=False, cent_type='riemann_mean', if_plot=False):
    if tl_bool == True:
        train_nontarget_cov, train_target_cov, test_nontarget_cov, test_target_cov = transfer_learn(train_nontarget_cov, train_target_cov, test_nontarget_cov, test_target_cov, cent_type)

    x_train = np.vstack([train_nontarget_cov, train_target_cov])
    y_train = np.hstack([np.zeros(np.size(train_nontarget_cov, 0)), np.ones(np.size(train_target_cov, 0))])
    x_test = np.vstack([test_nontarget_cov, test_target_cov])
    y_test = np.hstack([np.zeros(np.size(test_nontarget_cov, 0)), np.ones(np.size(test_target_cov, 0))])

    if cent_type == 'riemann_mean':
        cent0 = riemannian_mean(train_nontarget_cov)
        cent1 = riemannian_mean(train_target_cov)
    elif cent_type == 'euclid_mean':
        cent0 = euclidean_mean(train_nontarget_cov)
        cent1 = euclidean_mean(train_target_cov)
    elif cent_type == 'mat_med':
        cent0 = matrix_median(train_nontarget_cov)
        cent1 = matrix_median(train_target_cov)
    elif cent_type == 'riemann_med':
        cent0 = riemannian_median(train_nontarget_cov)
        cent1 = riemannian_median(train_target_cov)
    elif cent_type == 'huber':
        cent0 = huber_centroid(train_nontarget_cov)
        cent1 = huber_centroid(train_target_cov)
    else:
        raise ValueError("No such center type. Options are: riemann_mean, euclid_mean, mat_med, riemann_med, and huber")

    res = mdm_riemann_classifier(cent0, cent1, x_test, y_test)

    n_nontarget_test = len(test_nontarget_cov)
    n_target_test = len(test_target_cov)
    confus = get_confusion_mat(res, y_test, n_nontarget_test, n_target_test, if_plot)
    stats = get_stats(confus)
    return stats

def chance_perf(n_nontarget, n_target):
    n_trials = n_nontarget + n_target
    prob_nontarget = n_nontarget / n_trials
    prob_target = n_target / n_trials

    n, p = 1, prob_target
    chance_pred = np.random.binomial(n, p, n_trials)

    return chance_pred
        