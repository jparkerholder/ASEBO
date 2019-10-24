import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import cholesky
from numpy.linalg import LinAlgError
from numpy.random import standard_normal

from worker import worker

def aggregate_rollouts(master, A, params, n_samples):
    
    all_rollouts = np.zeros([n_samples, 2])

    timesteps = 0
    
    for i in range(n_samples):
        w = worker(params, master, A, i)
        all_rollouts[i] = np.reshape(w.do_rollouts(), 2)
        timesteps += w.timesteps

    all_rollouts = (all_rollouts - np.mean(all_rollouts)) / (np.std(all_rollouts)  + 1e-8)
    
    m = np.array(all_rollouts[:, 0] - all_rollouts[:, 1])
    return(m, timesteps)

def ES(params, master, G):
        
    if params['n_iter'] >= params['k']:
        pca = PCA()
        pca_fit = pca.fit(G)
        var_exp = pca_fit.explained_variance_ratio_
        var_exp = np.cumsum(var_exp)
        n_samples = np.argmax(var_exp > params['threshold']) + 1
        if n_samples < params['min']:
            n_samples = params['min']
        U = pca_fit.components_[:n_samples]
        UUT = np.matmul(U.T, U)
        U_ort = pca_fit.components_[n_samples:]
        UUT_ort = np.matmul(U_ort.T, U_ort)
        alpha = params['alpha']
        if params['n_iter'] == params['k']:
            n_samples = params['num_sensings']
    else:
        UUT = np.zeros([master.N, master.N])
        alpha = 1
        n_samples = params['num_sensings']
    
    np.random.seed(None)
    cov = (alpha/master.N) * np.eye(master.N) + ((1-alpha) / n_samples) * UUT
    cov *= params['sigma']
    mu = np.repeat(0, master.N)
    #A = np.random.multivariate_normal(mu, cov, n_samples)
    A = np.zeros((n_samples, master.N))
    try:
        l = cholesky(cov, check_finite=False, overwrite_a=True)
        for i in range(n_samples):
            try:
                A[i] = np.zeros(master.N) + l.dot(standard_normal(master.N))
            except LinAlgError:
                A[i] = np.random.randn(master.N)
    except LinAlgError:
        for i in range(n_samples):
            A[i] = np.random.randn(master.N)  
    A /= np.linalg.norm(A, axis =-1)[:, np.newaxis]
        
    m, timesteps = aggregate_rollouts(master, A, params, n_samples)
    
    g = np.zeros(master.N)
    for i in range(n_samples):
        eps = A[i, :]
        g += eps * m[i]
    g /= (2 * params['sigma'])
    
    if params['n_iter'] >= params['k']:
        params['alpha'] = np.linalg.norm(np.dot(g, UUT_ort))/np.linalg.norm(np.dot(g, UUT))
    
    return(g, n_samples, timesteps)

