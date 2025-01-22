import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


X,y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))


def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return W,b

def model(X,W,b):
    Z = X.dot(W)+b
    A = 1/(1+np.exp(-Z))
    return A

def log_loss(A,y):
    return 1/len(y) * np.sum(-y*np.log(A) - (1-y)*np.log(1-A))


W,b = initialisation(X)
A = model(X,W,b)
L = log_loss(A,y) 

assert(X.shape == (100, 2))
assert(y.shape == (100, 1))
assert(A.shape == (100, 1))
print("log loss:", L)
plt.scatter(X[:,0], X[:,1], c=y, cmap="summer")
plt.show()
