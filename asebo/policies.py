import numpy as np
from scipy.linalg import toeplitz
import gym
from copy import copy
# Toeplitz policy from Choromanski (2018)
# Can only have 2 layers

class ToeplitzPolicy(object):
    
    def __init__(self, policy_params):
        
        self.init_seed = policy_params['seed']
        self.ob_dim = policy_params['ob_dim']
        self.h_dim = policy_params['h_dim']
        self.ac_dim = policy_params['ac_dim']
        
        self.w1 = self.weight_init(self.ob_dim + self.h_dim -1, policy_params['zeros'])
        self.w2 = self.weight_init(self.h_dim * 2 - 1, policy_params['zeros'])
        self.w3 = self.weight_init(self.ac_dim + self.h_dim - 1, policy_params['zeros'])
        
        self.W1 = self.build_layer(self.h_dim, self.ob_dim, self.w1)
        self.W2 = self.build_layer(self.h_dim, self.h_dim, self.w2)
        self.W3 = self.build_layer(self.ac_dim, self.h_dim, self.w3)
        
        self.b1 = self.weight_init(self.h_dim, policy_params['zeros'])
        self.b2 = self.weight_init(self.h_dim, policy_params['zeros'])
    
        self.params = np.concatenate([self.w1, self.b1, self.w2, self.b2, self.w3])
        self.N = len(self.params)
    
    def weight_init(self, d, zeros):
        
        if zeros:
            w = np.zeros(d)
        else:
            np.random.seed(self.init_seed)
            w = np.random.rand(d) / np.sqrt(d)
        return(w)
    
    def build_layer(self, d1, d2, v):
        # len v = d1 + d2 - 1
        col = v[:d1]
        row = v[(d1-1):]
        
        W = toeplitz(col, row)
        return(W)
    
    def update(self, vec):
        
        self.params += vec
        
        self.w1 += vec[:len(self.w1)]
        vec = vec[len(self.w1):]

        self.b1 += vec[:len(self.b1)]
        vec = vec[len(self.b1):]
        
        self.w2 += vec[:len(self.w2)]
        vec = vec[len(self.w2):]

        self.b2 += vec[:len(self.b2)]
        vec = vec[len(self.b2):]

        self.w3 += vec
        
        self.W1 = self.build_layer(self.h_dim, self.ob_dim, self.w1)
        self.W2 = self.build_layer(self.h_dim, self.h_dim, self.w2)
        self.W3 = self.build_layer(self.ac_dim, self.h_dim, self.w3)
        
    def evaluate(self, X):
        
        #if len(X.shape) == 1:
        #    X = X.reshape(X.shape[0], 1)
        
        z1 = np.tanh(np.dot(self.W1, X) + self.b1)
        z2 = np.tanh(np.dot(self.W2, z1) + self.b2)
        return(np.tanh(np.dot(self.W3, z2)))


class LinearPolicy(object):
    
    def __init__(self, policy_params):
        
        self.init_seed = policy_params['seed']
        self.ob_dim = policy_params['ob_dim']
        self.h_dim = policy_params['h_dim']
        self.ac_dim = policy_params['ac_dim']
        
        self.w = self.weight_init(self.ob_dim * self.ac_dim, policy_params['zeros'])
        self.W = self.w.reshape(self.ac_dim, self.ob_dim)

        self.params = copy(self.w)
        self.N = len(self.params)
    
    def weight_init(self, d, zeros):
        
        if zeros:
            w = np.zeros(d)
        else:
            np.random.seed(self.init_seed)
            w = np.random.rand(d) / np.sqrt(d)
        return(w)
    
    def update(self, vec):
        
        
        self.w += vec
        self.W = self.w.reshape(self.ac_dim, self.ob_dim)
        
        self.params = copy(self.w)

    def evaluate(self, X):
        
        X = X.reshape(X.size, 1)

        return(np.tanh(np.dot(self.W, X)))