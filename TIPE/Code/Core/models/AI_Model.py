
__init__ = "AI_Model"

import pickle
from tqdm import tqdm 
import numpy as np
from sklearn.metrics import accuracy_score, r2_score

from ..utils.utils import courbe_perf


class Resaux:
    def __init__(self, X=None, y=None, X_test=None, y_test=None, nb_neurone_couche=[1], learning_rate=0.1, nb_iter=1, path=None, threshold_val=0.5, qt=None):
        """ Initialise le réseau de neurones """
        self.path = path
        self.W = None
        self.b = None
        self.L = []
        self.L_t = []  
        self.acc = []
        self.acc_t = []
        self.threshold_val = threshold_val

        self.nb_classes = nb_neurone_couche[-1]  
        self.nb_neurone_couche = nb_neurone_couche

        self.qt = qt
        self.is_regression = self.qt is not None and self.nb_classes == 1

        if self.qt is not None and nb_neurone_couche[-1] != 1:
            raise ValueError("Régression autorisée uniquement si la dernière couche contient 1 neurone (nb_classes == 1)")

        if X is not None and y is not None:
            self.W = [np.random.randn(nb_neurone_couche[0], X.shape[1])]
            self.W += [np.random.randn(nb_neurone_couche[i], nb_neurone_couche[i - 1]) for i in range(1, len(nb_neurone_couche))]
            self.b = [np.random.randn(n, 1) for n in nb_neurone_couche]
            self.train(X, y, X_test, y_test, learning_rate, nb_iter)

    def MSE(self, A, y):
        """ Calcule l'erreur quadratique moyenne """
        return np.mean((A[-1].flatten() - y.flatten()) ** 2)   
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def log_loss(self, A, y):
            epsilon = 1e-15
            A = np.clip(A[-1], epsilon, 1 - epsilon)
            return -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))
    
    def forward_propagation(self, X):
        """ Calcule la propagation avant du réseau de neurones """
        Z = [np.dot(self.W[0], X.T) + self.b[0]]
        A = [1 / (1 + np.exp(-Z[0]))]  # Activation sigmoïde pour la première couche

        for i in range(1, len(self.W) - 1):
            Z.append(self.W[i].dot(A[i - 1]) + self.b[i])
            A.append(1 / (1 + np.exp(-Z[i])))

        # Dernière couche
        Z_last = self.W[-1].dot(A[-1]) + self.b[-1]
        Z.append(Z_last)

        if self.is_regression:
            A.append(Z_last)  # Pas d'activation en sortie pour la régression
        elif self.nb_classes == 1:
            A.append(1 / (1 + np.exp(-Z_last)))  # Sigmoïde pour classification binaire
        else:
            A.append(self.softmax(Z_last))  # Softmax pour classification multiclasse

        return A
    
    def back_propagation(self, A, X, y):
        """ Calcule les gradients de la fonction de perte par rapport aux poids et biais """
        m = X.shape[0]
        dW = []
        db = []

        if self.is_regression:
            # y shape = (m,) → reshape to (1, m)
            dZ = A[-1] - y.reshape(1, -1)
        elif self.nb_classes == 1:
            y = y.reshape(1, -1)
            dZ = A[-1] - y
        else:
            y_one_hot = np.zeros_like(A[-1])
            y_one_hot[y, np.arange(m)] = 1
            dZ = A[-1] - y_one_hot

        # Backpropagation générique
        for i in reversed(range(len(self.W))):
            A_prev = A[i - 1] if i > 0 else X.T

            dW_i = np.dot(dZ, A_prev.T) / m
            db_i = np.sum(dZ, axis=1, keepdims=True) / m

            dW.insert(0, dW_i)
            db.insert(0, db_i)

            if i > 0:
                dA_prev = np.dot(self.W[i].T, dZ)
                dZ = dA_prev * A[i - 1] * (1 - A[i - 1])  # Dérivée de sigmoïde

        return dW, db

    def cross_entropy_loss(self, A, y):
        m = y.shape[0]
        probs = np.clip(A[-1], 1e-15, 1 - 1e-15)

        if self.nb_classes != probs.shape[0]:
            raise ValueError(f"Nombre de classes incohérent : self.nb_classes={self.nb_classes}, probs.shape={probs.shape}")

        y_one_hot = np.zeros_like(probs)
        y_one_hot[y, np.arange(m)] = 1

        return -np.sum(y_one_hot * np.log(probs)) / m

    def loss(self, A, y):
        """ Calcule la fonction de perte logistique """
        if self.is_regression:
            return self.MSE(A, y)
        elif self.nb_classes == 1:
            return self.log_loss(A, y)
        else:
            return self.cross_entropy_loss(A, y)

    def update(self, dW, db, learning_rate):
        """ Met à jour les poids et le biais """
        '''for i in range(len(self.W)):
            print(f"dW[{i}].shape", dW[i].shape)
            print(f"self.W[{i}].shape", self.W[i].shape)
            print(f"db[{i}].shape", db[i].shape)
            print(f"self.b[{i}].shape", self.b[i].shape)
        '''
        for i in range(len(self.W)):
            self.W[i] -= learning_rate * dW[i]
            self.b[i] -= learning_rate * db[i]
    
    def predict(self, X):
        """Prédiction sur de nouvelles données, toujours en (n_samples, n_features)."""
        # Cas vecteur 1D (un seul exemple) → on le passe en shape (1, n_features)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        A = self.forward_propagation(X)
        out = A[-1]
        if self.is_regression:
            return self.qt.inverse_transform(out.T).flatten()
        elif self.nb_classes == 1:
            return (out.flatten() >= self.threshold_val).astype(int)
        else:
            # Multi-classes : out.shape == (nb_classes, n_samples)
            return np.argmax(out, axis=0)

    def evaluate_metrics(self, y, y_pred, y_test, y_pred_test):
        if self.is_regression:
            y = self.qt.inverse_transform(y.reshape(-1, 1)).flatten()
            y_test  = self.qt.inverse_transform(y_test.reshape(-1, 1)).flatten()
            self.acc.append(r2_score(y, y_pred))
            self.acc_t.append(r2_score(y_test, y_pred_test))
        elif self.nb_classes == 1:
            self.acc.append(accuracy_score((y >= self.threshold_val).astype(int), y_pred))
            self.acc_t.append(accuracy_score((y_test >= self.threshold_val).astype(int), y_pred_test))
        else:
            self.acc.append(accuracy_score(y.flatten().astype(int), y_pred))
            self.acc_t.append(accuracy_score(y_test.flatten().astype(int), y_pred_test))

    def train(self, X, y, X_test, y_test, learning_rate=1e-2, nb_iter=10000, partialsteps=10):
        """ Entraîne le modèle sur les données d'entraînement """
        for i in tqdm(range(nb_iter)):
            A = self.forward_propagation(X)

            if i % partialsteps == 0:
                self.L.append(self.loss(A, y))
                self.L_t.append(self.loss(self.forward_propagation(X_test), y_test))
                y_pred_train = self.predict(X)
                y_pred_test = self.predict(X_test)
                self.evaluate_metrics(y, y_pred_train, y_test, y_pred_test)

            if i % (partialsteps*100) == 0:
                if self.path is not None:
                    self.save(self.path, bool_p=False)
            
            dW, db = self.back_propagation(A, X, y)
            self.update(dW, db, learning_rate)  

    def get_model_type(self):
        if self.is_regression:
            return "regression"
        elif self.nb_classes == 1:
            return "binaire"
        else:
            return "multiclass"

    def save(self, filename, bool_p=True):
        """ Sauvegarde du modèle dans un fichier """
        with open(filename, 'wb') as f:
            pickle.dump({
                'W': self.W,
                'b': self.b,
                'L': self.L,
                'L_t': self.L_t,
                'acc': self.acc,
                'acc_t': self.acc_t,
                'nb_neurone_couche': self.nb_neurone_couche,
                'path': self.path,
                'threshold_val': self.threshold_val,
                'nb_classes': self.nb_classes,
                'qt': self.qt,
                'is_regression': self.is_regression
            }, f)
        courbe_perf(self, self.path.replace(".pkl", ".png").replace("save", "curve").replace("TIPE/Code/Saves/", "TIPE/Code/Saves_Curves/"), bool_p)
        if bool_p:
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
            self.nb_neurone_couche = data['nb_neurone_couche']
            self.path = data['path']
            self.threshold_val = data['threshold_val']
            self.nb_classes = data['nb_classes']
            self.qt = data['qt'] if 'qt' in data else None
            self.is_regression = data['is_regression'] if 'is_regression' in data else False
        print(f"Modèle chargé depuis {filename}")
