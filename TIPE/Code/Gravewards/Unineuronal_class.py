import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class Neurone: 

    def __init__(self, X=None, y=None, X_test=None, y_test=None, learning_rate=0.1, nb_iter=1):
        """ Initialise le neurone """
        self.W = None
        self.b = None
        self.L = []
        self.L_t = []
        self.acc = []
        self.acc_t = []

        if X is not None and y is not None:
            self.W = np.random.randn(X.shape[1], 1)
            self.b = np.random.randn(1)
            self.train(X, y, X_test, y_test, learning_rate, nb_iter)

    def model(self, X):
        """ Calcule la sortie du neurone """
        Z = X.dot(self.W) + self.b
        A = 1 / (1 + np.exp(-Z)) 
        return A

    def log_loss(self, A, y):
        """ Calcule la fonction de perte logistique """
        epsilon = 1e-15
        A = np.clip(A, epsilon, 1 - epsilon)  # Ensure A is within (0, 1)
        return -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))

    def gradients(self, A, X, y):
        """ Calcule les gradients de la fonction de perte par rapport aux poids et au biais """
        dW = np.dot(X.T, A - y) / len(y)
        db = np.sum(A - y, axis=0) / len(y)
        return dW, db

    def update(self, dW, db, learning_rate):
        """ Met à jour les poids et le biais """
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

    def train(self, X, y, X_test, y_test, learning_rate=1e-2, nb_iter=10000, partialsteps=100):
        """ Entraîne le modèle sur les données d'entraînement """
        for i in tqdm(range(nb_iter)):
            A = self.model(X)

            if i % partialsteps == 0:
                self.L.append(self.log_loss(A, y))
                self.L_t.append(self.log_loss(self.model(X_test), y_test))
                self.acc.append(accuracy_score(y >= 0.5, self.predict(X)))
                self.acc_t.append(accuracy_score(y_test >= 0.5, self.predict(X_test)))

            dW, db = self.gradients(A, X, y)
            self.update(dW, db, learning_rate)

    def predict(self, X):
        """ Prédiction sur de nouvelles données """
        A = self.model(X)
        return A >= 0.5

    def save(self, filename):
        """ Sauvegarde du modèle dans un fichier """
        with open(filename, 'wb') as f:
            pickle.dump({'W': self.W, 'b': self.b, 'L': self.L, 'L_t': self.L_t, 'acc': self.acc, 'acc_t': self.acc_t}, f)
        print(f"Modèle sauvegardé dans {filename}")

    def load(self, filename):
        """ Charge un modèle depuis un fichier """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.W = data['W']
            self.b = data['b']
            self.L = data['L']
            self.L_t = data['L_t']
            self.acc = data['acc']
            self.acc_t = data['acc_t']
        print(f"Modèle chargé depuis {filename}")
