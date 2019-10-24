
import numpy as np

def Adam(dx, m, v, learning_rate, t, eps = 1e-8, beta1 = 0.9, beta2 = 0.999):
    m = beta1 * m + (1 - beta1) * dx
    mt = m / (1 - beta1 ** t)
    v = beta2 * v + (1-beta2) * (dx **2)
    vt = v / (1 - beta2 ** t)
    update = learning_rate * mt / (np.sqrt(vt) + eps)
    return(update, m, v)