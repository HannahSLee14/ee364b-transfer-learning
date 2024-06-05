import mat73
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pyriemann.utils.base import sqrtm, invsqrtm, logm, expm, powm

# definition from https://ieeexplore.ieee.org/document/8013808
def riemannian_distance(X,Y):
    X_invsqrtm = invsqrtm(X)
    if X.ndim > 2 or Y.ndim > 2:
        return np.sum(np.log(np.linalg.eigvalsh(X_invsqrtm @ Y @ X_invsqrtm))**2, axis=1)
    else:
        return np.sum(np.log(np.linalg.eigvalsh(X_invsqrtm @ Y @ X_invsqrtm))**2)

# matches what we expect: https://ieeexplore.ieee.org/document/8013808
def euclidean_mean(P_set):
    return np.mean(P_set, axis=0)

def grad_riemann_mean(P_set, P_mean, weights):
    P_invsqrt = invsqrtm(P_mean)
    return np.einsum('a,abc->bc', weights, logm(P_invsqrt @ P_set @ P_invsqrt))

# modified from: https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/utils/mean.py#L22 
# derived from: https://link.springer.com/chapter/10.1007/978-3-642-00826-9_6 
# find riemannian mean using gradient descent 
def riemannian_mean(P_set, max_iter=50, nu=1.0, tol=10e-9, weights=None):
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max

    mean_curr = euclidean_mean(P_set)
    n_trials, n_cov, _ = P_set.shape

    if weights == None:
        weights = np.ones(n_trials) / n_trials # evenly weigh all trials

    for i in range(max_iter):
        mean_sqrt = sqrtm(mean_curr)
        grad = grad_riemann_mean(P_set, mean_curr, weights)
        mean_curr = mean_sqrt @ expm(nu * grad) @ mean_sqrt

        # this is taken directly from the first link
        crit = np.linalg.norm(grad, ord='fro')
        h = nu * crit
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu
        if crit <= tol or nu <= tol:
            break

    return mean_curr

# stopping criteria from: https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/utils/mean.py#L22
# from: https://proceedings.neurips.cc/paper/2017/hash/6ef80bb237adf4b6f77d0700e1255907-Abstract.html
# according to the paper, it's supposed to be accelerated - but at least with this implementation, it is slower by an order of magnitude
# speed changes depending on L (not linearly) - out of scope of this project to investigate this further
def momen_riemannian_mean(P_set, max_iter=50, nu=1.0, tol=10e-9, weights=None):
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max

    x_prev = euclidean_mean(P_set)
    x = euclidean_mean(P_set)
    y = euclidean_mean(P_set)
    n_trials, n_cov, _ = P_set.shape

    mu = 2*n_trials
    L = 2.25*n_trials
    gamma = 1 / (1-np.sqrt(mu/L))
    cons = np.sqrt(1-np.sqrt(mu/L))
    beta = (4 / np.sqrt(mu*L)) - (1/L)

    if weights == None:
        weights = np.ones(n_trials) / n_trials # evenly weigh all trials
    
    for i in range(max_iter):
        grad = grad_riemann_mean(P_set, y, weights)

        # exponential mapping
        y_sqrt = sqrtm(y)
        y_invsqrt = invsqrtm(y)
        x_prev = x
        x = y_sqrt @ expm(nu*grad) @ y_sqrt #y_sqrt @ expm(y_invsqrt @ (-nu * grad) @ y_invsqrt) @ y_sqrt

        # solve for x 
        x_sqrt = sqrtm(x)
        y = x_sqrt @ np.linalg.inv(expm(cons * logm(y_invsqrt @ x_prev @ y_invsqrt) + (gamma*beta)*grad_riemann_mean(P_set, y, weights))) @ x_sqrt

        # this is taken directly from the first link
        crit = np.linalg.norm(grad, ord='fro')
        h = nu * crit
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu
        if crit <= tol or nu <= tol:
            break
    
    return x 

# from https://www.sciencedirect.com/science/article/pii/S0377042711005218
# uses ADMM + proximal update to do the update
def matrix_median(P_set, lam=0, gamma=1, max_iter=200, tol=10e-9):
    n_trials, n_cov, _ = P_set.shape

    mat_rn = np.zeros((n_cov, n_cov))
    mat_rn = mat_rn.T @ mat_rn

    V_curr = np.tile(mat_rn, (n_trials,1,1)) 

    mat_rn = np.zeros((n_cov, n_cov)) 
    mat_rn = mat_rn.T @ mat_rn
    B_curr = np.tile(mat_rn, (n_trials,1,1))
    
    mat_rn = np.zeros((n_cov, n_cov)) 
    mat_rn = mat_rn.T @ mat_rn
    S_curr = np.tile(mat_rn, (n_trials,1,1))

    X_curr = euclidean_mean(P_set) # informed start
    X_prev = X_curr

    for i in range(max_iter):
        # X update
        X_prev = X_curr
        X_curr = np.linalg.inv((lam*gamma + n_trials)*np.identity(n_cov)) @ (np.sum(V_curr - B_curr, axis=0))

        # termination condition
        if np.linalg.norm(np.abs(X_curr - X_prev), ord='fro') < tol:
            break

        # V update through proximal update on Y
        Y_curr = V_curr - P_set
        S_curr = B_curr + np.tile(X_curr, (n_trials,1,1)) - P_set
        S_norm = np.linalg.norm(S_curr, axis=(1,2), ord='fro')
        Y_new = np.zeros(Y_curr.shape)
        prox_mult = np.tile((1 - (gamma/S_norm)), (n_cov, n_cov, 1))
        prox_mult = np.moveaxis(prox_mult,-1,0)
        greater_idx = S_norm >= gamma
        Y_new[greater_idx] = prox_mult[greater_idx] * S_curr[greater_idx]
        Y_curr = Y_new  
        V_curr = Y_curr + P_set # update V_curr

        # B update - dual update
        B_curr = B_curr + np.tile(X_curr, (n_trials,1,1)) - V_curr

    return X_curr 

# See https://www.sciencedirect.com/science/article/pii/S1053811908012019?via%3Dihub 
# uses steepest descent
# directly from: https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/utils/median.py
def riemannian_median(P_set, nu=1, max_iter=50, tol=10e-9, weights=None):
    n_trials, n_cov, _ = P_set.shape

    curr_med = euclidean_mean(P_set)

    if weights == None:
        weights = np.ones(n_trials) / n_trials # evenly weigh all trials

    for i in range(max_iter):
        distances = np.array([riemannian_distance(P_indiv, curr_med) for P_indiv in P_set])
        is_nonzero = (~(distances == 0))
        nonzero_weights = weights[is_nonzero] / distances[is_nonzero]

        med_sqrt = sqrtm(curr_med)
        med_invsqrt = invsqrtm(curr_med)
        tangent_vecs = logm(med_invsqrt @ P_set[is_nonzero] @ med_invsqrt)
        grad = np.einsum('a,abc->bc', nonzero_weights / np.sum(nonzero_weights), tangent_vecs)
        curr_med = med_sqrt @ expm(nu * grad) @ med_sqrt

        crit = np.linalg.norm(grad, ord='fro')
        if crit <= tol:
            break

    return curr_med

def log_map(curr_cent, mat):
    cent_invsqrt = invsqrtm(curr_cent)
    cent_sqrt = sqrtm(curr_cent)
    return cent_sqrt @ logm(cent_invsqrt @ mat @ cent_invsqrt) @ cent_sqrt

def exp_map(curr_cent, mat):
    cent_invsqrt = invsqrtm(curr_cent)
    cent_sqrt = sqrtm(curr_cent)
    return cent_sqrt @ expm(cent_invsqrt @ mat @ cent_invsqrt) @ cent_sqrt

# threshold based outlier (1.5*third quartile)
def huber_grad(P_set, curr_cent):
    n_trials, n_cov, _ = P_set.shape
    dists = riemannian_distance(P_set, curr_cent)
    third_quart = np.quantile(dists, 0.75)
    T = 1.5*third_quart
    idx_outlier = (dists > T)

    dists = np.tile(dists, [n_cov,n_cov,1])
    dists = np.moveaxis(dists,-1,0)

    return (-2/n_trials)*np.sum(log_map(curr_cent, P_set)[~idx_outlier,:], axis=0) - (T/n_trials)*np.sum(log_map(curr_cent, P_set)[idx_outlier,:]/dists[idx_outlier,:], axis=0)

def huber_obj(P_set, curr_cent):
    n_trials, n_cov, _ = P_set.shape
    dists = riemannian_distance(P_set, curr_cent)
    third_quart = np.quantile(dists, 0.75)
    T = 1.5*third_quart
    idx_outlier = (dists > T)

    return (1/n_trials)*np.sum(dists[~idx_outlier]) + (T/n_trials)*np.sum(np.sqrt(dists[idx_outlier]))

# From: https://ieeexplore.ieee.org/abstract/document/7523317
def huber_centroid(P_set, alpha=0.25, mu=0.5, nu_init=0.5, max_iter=50, tol=10e-9): 
    n_trials, n_cov, _ = P_set.shape
    curr = euclidean_mean(P_set)
    nu = nu_init

    for i in range(max_iter):
        grad = huber_grad(P_set, curr)

        # attempt at Armijo backsearching - need to check update
        # while huber_obj(P_set, exp_map(curr, -nu*grad)) > (huber_obj(P_set, curr) + nu*np.sum(huber_grad(P_set,curr)*exp_map(curr, -grad))):
        #     nu = mu * nu

        curr = exp_map(curr, -nu*grad)
        crit = np.linalg.norm(grad, ord='fro')
        if crit <= tol:
            break

    return curr