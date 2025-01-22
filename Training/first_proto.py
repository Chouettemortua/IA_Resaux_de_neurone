import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# initialisation de vecteur contenant le donné X et la reférence y
X,y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
# ici ce ne sont pas des vrais donné et on a initialiser un vecteur (100,2) 
# alors que les réponse sont un vecteur (100,1) donc ont garde que la première colone
y = y.reshape((y.shape[0], 1))


class Neurone: 
    # Sert a l'initialisation du vecteur des paramétre sur les entrée
    # et la constante
    def __init__(self,X,y):
        self.W = np.random.randn(X.shape[1], 1)
        self.b = np.random.randn(1)
        self.L = []
        self.neurone(X,y)

    # definit le modèle donc la fonction linéaire
    # puis transforme la sortie linéaire en un pourcentage
    def model(self,X):
        Z = X.dot(self.W)+self.b
        A = 1/(1+np.exp(-Z))
        return A

    #On definit la fonction de coût donc log loss ici
    def log_loss(self,A,y):
        return 1/len(y) * np.sum(-y*np.log(A) - (1-y)*np.log(1-A))

    #On definit nos gradient
    def gradients(self,A,X,y):
        dW = 1/len(y) * np.dot(X.T,A-y)
        db = 1/len(y) * np.sum(A-y)
        return dW,db

    #On definit la decente de gradient pour ajuster les paramètre
    def update(self,dW, db, learning_rate):
        self.W = self.W-learning_rate*dW
        self.b = self.b-learning_rate*db

    def neurone(self,X, y,learning_rate=0.1, nb_iter=100):
        #boucle d'update
        for _ in range(nb_iter):
            A = self.model(X)
            self.L.append(self.log_loss(A, y))
            dW,db = self.gradients(A,X,y)
            self.update(dW, db, learning_rate)


"""#qq test basic
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
"""
neu = Neurone(X,y)

plt.plot(neu.L)
plt.show()

#plt.scatter(X[:,0], X[:,1], c=y, cmap="summer")
#plt.show()
