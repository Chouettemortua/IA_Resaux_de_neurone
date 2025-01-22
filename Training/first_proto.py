import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# initialisation de vecteur contenant le donné X et la reférence y
X,y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
# ici ce ne sont pas des vrais donné et on a initialiser un vecteur (100,2) 
# alors que les réponse sont un vecteur (100,1) donc ont garde que la première colone
y = y.reshape((y.shape[0], 1))


# Sert a l'initialisation du vecteur des paramétre sur les entrée
# et la constante
def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return W,b

# definit le modèle donc la fonction linéaire
# puis transforme la sortie linéaire en un pourcentage
def model(X,W,b):
    Z = X.dot(W)+b
    A = 1/(1+np.exp(-Z))
    return A

#On definit la fonction de coût donc log loss ici
def log_loss(A,y):
    return 1/len(y) * np.sum(-y*np.log(A) - (1-y)*np.log(1-A))

#On definit nos gradient
def gradients(A, X, y):
    dW = 1/len(y) * np.dot(X.T,A-y)
    db = 1/len(y) * np.sum(A-y)
    return dW,db

#On definit la decente de gradient pour ajuster les paramètre
def update(dW, db, W, b, learning_rate):
    W = W-learning_rate*dW
    b = b-learning_rate*db
    return W,b


#qq test basic
W,b = initialisation(X)
A = model(X,W,b)
L = log_loss(A,y) 
dW,db = gradients(A,X,y)
W,b = update(dW, db, W, b, 2)


assert(X.shape == (100, 2)) 
assert(y.shape == (100, 1))
assert(A.shape == (100, 1))
assert(dW.shape == (2, 1))
assert(db.shape == ())
print("log loss:", L)


#plt.scatter(X[:,0], X[:,1], c=y, cmap="summer")
#plt.show()
