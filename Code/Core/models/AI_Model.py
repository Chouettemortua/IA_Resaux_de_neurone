
__init__ = "AI_Model"

import pickle
import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from PyQt6.QtCore import QObject, pyqtSignal

from ..utils.utils import courbe_perf


class Resaux(QObject):
    '''Classe de géneration et d'entrainement d'un réseau de neurones MLP (regression, binaire, multiclass) lié à une interface PyQt6'''

    progress_updated = pyqtSignal(int) # Signal pour mettre à jour la barre de progression dans l'interface PyQt6
    curve_save = pyqtSignal() # Signal de la sauvegarde des courbe

    def __init__(self, X=None, y=None, X_test=None, y_test=None, nb_neurone_couche=[1], path=None, threshold_val=0.5, qt=None):
        """ Initialise le réseau de neurones avec une architecture donnée et des poids aléatoires .
        Args:
            X (np.ndarray): Données d'entrée d'entraînement de forme (n_samples, n_features).
            y (np.ndarray): Étiquettes de sortie d'entraînement de forme (n_samples,) ou (n_samples, n_classes) pour la classification multiclasse.
            X_test (np.ndarray): Données d'entrée de test de forme (n_samples, n_features).
            y_test (np.ndarray): Étiquettes de sortie de test de forme (n_samples,) ou (n_samples, n_classes) pour la classification multiclasse.
            nb_neurone_couche (list): Liste contenant le nombre de neurones par couche, incluant la couche d'entrée et la couche de sortie.
            path (str): Chemin pour sauvegarder le modèle entraîné.
            threshold_val (float): Seuil pour la classification binaire.
            qt: Objet de transformation inverse pour les tâches de régression.
        """

        super().__init__()

        self.path = path
        self.W = None # Poids
        self.b = None # Biais
        # Liste des pertes et des précisions
        self.L = []  
        self.L_t = []  
        self.acc = []
        self.acc_t = []

        self.partialsteps = None  # Fréquence de sauvegarde du modèle et d'évaluation des métriques (régler lors du premier appel de train)
        self.threshold_val = threshold_val

        self.nb_classes = nb_neurone_couche[-1]  
        self.nb_neurone_couche = nb_neurone_couche

        self.qt = qt
        self.is_regression = self.qt is not None and self.nb_classes == 1

        if self.qt is not None and nb_neurone_couche[-1] != 1:
            raise ValueError("Régression autorisée uniquement si la dernière couche contient 1 neurone.")

        # Initialisation des poids et biais
        if X is not None and y is not None:
            self.init_weights(X)      

    def summary(self):
        """ Affiche un résumé de l'architecture du réseau de neurones. """
        print("Résumé du modèle de réseau de neurones :")
        print("--------------------------------------------------")
        print(f"{'Couche':<10} {'Neurones':<10} {'Poids':<20} {'Biais':<20}")
        print("--------------------------------------------------")
        for i, (n_neurons, W, b) in enumerate(zip(self.nb_neurone_couche, self.W, self.b)):
            print(f"{i + 1:<10} {n_neurons:<10} {str(W.shape):<20} {str(b.shape):<20}")
        print("--------------------------------------------------")
        model_type = self.get_model_type()
        print(f"Type de modèle : {model_type}")
        if model_type == "binaire":
            print(f"Seuil de classification binaire : {self.threshold_val}")
        print("--------------------------------------------------")

    def random_weights(self, X):
        """ Initialise les poids et biais du réseau de neurones de manière aléatoire. (a remplacer par Glorot_initialization quand elle est corrigée)"""
        W = [np.random.randn(self.nb_neurone_couche[0], X.shape[1])]
        W += [np.random.randn(self.nb_neurone_couche[i], self.nb_neurone_couche[i - 1]) for i in range(1, len(self.nb_neurone_couche))]
        b = [np.random.randn(n, 1) for n in self.nb_neurone_couche]
        return W, b

    def glorot_initialization(self, X):
        """ Initialise les poids et biais du réseau de neurones en utilisant l'initialisation de Glorot (Xavier).
        Args:
            X (np.ndarray): Données d'entrée de forme (n_samples, n_features).
        """
        W = []
        b = []
        #première couche
        limit = np.sqrt(6 / (X.shape[1] + self.nb_neurone_couche[0]))
        W.append(np.random.uniform(-limit, limit, (self.nb_neurone_couche[0], X.shape[1])))
        b.append(np.zeros((self.nb_neurone_couche[0], 1)))

        #couche caché et de sortie
        for i in range(1, len(self.nb_neurone_couche)):
            limit = np.sqrt(6 / (self.nb_neurone_couche[i - 1] + self.nb_neurone_couche[i]))
            W.append(np.random.uniform(-limit, limit, (self.nb_neurone_couche[i], self.nb_neurone_couche[i - 1])))
            b.append(np.zeros((self.nb_neurone_couche[i], 1)))

        return W, b

    def he_initialization(self, X):
        W = []
        b = []

        fan_in = X.shape[1]
        fan_out = self.nb_neurone_couche[0]
        limit = np.sqrt(2 / fan_in)
        W.append(np.random.randn(fan_out, fan_in) * limit)
        b.append(np.zeros((fan_out, 1)))

        for i in range(1, len(self.nb_neurone_couche)):
            fan_in = self.nb_neurone_couche[i - 1]
            fan_out = self.nb_neurone_couche[i]
            limit = np.sqrt(2 / fan_in)
            W.append(np.random.randn(fan_out, fan_in) * limit)
            b.append(np.zeros((fan_out, 1)))

        return W, b

    def init_weights(self, X):
        """ Initialise les poids et biais du réseau de neurones.
        Args:
            nb_neurone_couche (list): Liste contenant le nombre de neurones par couche, incluant la couche d'entrée et la couche de sortie.
            X (np.ndarray): Données d'entrée de forme (n_samples, n_features).
        """
        self.W, self.b = self.he_initialization(X)

    def MSE(self, A, y):
        """ Calcule l'erreur quadratique moyenne 
        Args:
            A (list): Liste des activations de chaque couche.
            y (np.ndarray): Étiquettes de sortie réelles.
        Returns:
            float: Valeur de l'erreur quadratique moyenne."""
        return np.mean((A[-1].flatten() - y.flatten()) ** 2)   

    def cross_entropy_loss(self, A, y):
        """ Calcule la fonction de perte par entropie croisée pour la classification multiclasse 
        Args:
            A (list): Liste des activations de chaque couche.
            y (np.ndarray): Étiquettes de sortie réelles.
            Returns:
                float: Valeur de la perte par entropie croisée."""
        m = y.shape[0]
        probs = np.clip(A[-1], 1e-15, 1 - 1e-15)

        if self.nb_classes != probs.shape[0]:
            raise ValueError(f"Nombre de classes incohérent : self.nb_classes={self.nb_classes}, probs.shape={probs.shape}")

        y_one_hot = np.zeros_like(probs)
        y_one_hot[y, np.arange(m)] = 1

        return -np.sum(y_one_hot * np.log(probs)) / m

    def log_loss(self, A, y):
        """ Calcule la fonction de perte logistique pour la classification binaire
        Args:
            A (list): Liste des activations de chaque couche.
            y (np.ndarray): Étiquettes de sortie réelles.
            Returns:
            float: Valeur de la perte logistique."""
        epsilon = 1e-15
        A = np.clip(A[-1], epsilon, 1 - epsilon)
        return -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))
    
    def loss(self, A, y):
        """ Calcule la fonction de perte adaptée selon le type de tâche (régression, binaire, multiclass)
        Args:
            A (list): Liste des activations de chaque couche.
            y (np.ndarray): Étiquettes de sortie réelles.
            Returns:
                float: Valeur de la fonction de perte."""
        if self.is_regression:
            return self.MSE(A, y)
        elif self.nb_classes == 1:
            return self.log_loss(A, y)
        else:
            return self.cross_entropy_loss(A, y)

    def softmax(self, z):
        """ Calcule la fonction softmax pour la classification multiclasse 
        Args:
            z (np.ndarray): Entrée de la couche de sortie.
            Returns:
            np.ndarray: Probabilités normalisées pour chaque classe."""
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward_propagation(self, X):
        """ Calcule la propagation avant du réseau de neurones en partant des données d'entrée X. 
            (en gros on calcule les activations de chaque couche puis on les stocke dans une liste A)
        Args:
            X (np.ndarray): Données d'entrée de forme (n_samples, n_features).
            Returns:
                list: Liste des activations de chaque couche."""
        Z = [np.dot(self.W[0], X.T) + self.b[0]]
        # A = [1 / (1 + np.exp(-Z[0]))]  # Activation sigmoïde pour la première couche
        A = [np.maximum(0, Z[0])]  # Activation ReLU pour la première couche

        for i in range(1, len(self.W) - 1):
            Z.append(self.W[i].dot(A[i - 1]) + self.b[i])
            #A.append(1 / (1 + np.exp(-Z[i]))) # Sigmoïde pour les couches cachées
            A.append(np.maximum(0, Z[i])) # ReLU pour les couches cachées

        # Dernière couche
        Z_last = self.W[-1].dot(A[-1]) + self.b[-1]
        Z.append(Z_last)

        if self.is_regression:
            A.append(Z_last)  # Pas d'activation en sortie pour la régression
        elif self.nb_classes == 1:
            A.append(1 / (1 + np.exp(-Z_last)))  # Sigmoïde pour classification binaire
        else:
            A.append(self.softmax(Z_last))  # Softmax pour classification multiclasse
            

        return A, Z
    
    def back_propagation(self, Z, A, X, y):
        """ Calcule les gradients de la fonction de perte par rapport aux poids et biais 
        Args:
            A (list): Liste des activations de chaque couche.
            X (np.ndarray): Données d'entrée de forme (n_samples, n_features).
            y (np.ndarray): Étiquettes de sortie réelles.
            Returns:
                tuple: Gradients des poids et des biais."""
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
                #dZ = dA_prev * A[i - 1] * (1 - A[i - 1])  # Dérivée de sigmoïde
                dZ = dA_prev * (Z[i - 1] > 0) # Dérivée de ReLU

        return dW, db

    def update(self, dW, db, learning_rate):
        """ Met à jour les poids et le biais 
        Args:
            dW (list): Gradients des poids.
            db (list): Gradients des biais.
            learning_rate (float): Taux d'apprentissage."""
        
        ''' test shapes
        for i in range(len(self.W)):
            print(f"dW[{i}].shape", dW[i].shape)
            print(f"self.W[{i}].shape", self.W[i].shape)
            print(f"db[{i}].shape", db[i].shape)
            print(f"self.b[{i}].shape", self.b[i].shape)
        '''
        for i in range(len(self.W)):
            self.W[i] -= learning_rate * dW[i]
            self.b[i] -= learning_rate * db[i]
    
    def predict(self, X):
        """Prédiction sur de nouvelles données, toujours en (n_samples, n_features).
        Args:
            X (np.ndarray): Données d'entrée de forme (n_samples, n_features).
            Returns:
                np.ndarray: Prédictions du modèle."""
        # Cas vecteur 1D (un seul exemple) → on le passe en shape (1, n_features)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        A, _ = self.forward_propagation(X)
        out = A[-1]
        if self.is_regression:
            return self.qt.inverse_transform(out.T).flatten()
        elif self.nb_classes == 1:
            return (out.flatten() >= self.threshold_val).astype(int)
        else:
            # Multi-classes : out.shape == (nb_classes, n_samples)
            return np.argmax(out, axis=0)

    def evaluate_metrics(self, y, y_pred, y_test, y_pred_test):
        """ Évalue les métriques de performance (précision ou R²) sur les données d'entraînement et de test et les stocke dans des liste interne au model.
        Args:
            y (np.ndarray): Étiquettes de sortie réelles d'entraînement.
            y_pred (np.ndarray): Prédictions du modèle sur les données d'entraînement.
            y_test (np.ndarray): Étiquettes de sortie réelles de test.
            y_pred_test (np.ndarray): Prédictions du modèle sur les données de test.
        """
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
    
    def train(self, X, y, X_test, y_test, curve_path, learning_rate=1e-2, nb_iter=10000, partialsteps=10):
        """ Entraîne le modèle sur les données d'entraînement 
        Args:
            X (np.ndarray): Données d'entrée d'entraînement de forme (n_samples, n_features).
            y (np.ndarray): Étiquettes de sortie d'entraînement de forme (n_samples,) ou (n_samples, n_classes) pour la classification multiclasse.
            X_test (np.ndarray): Données d'entrée de test de forme (n_samples, n_features).
            y_test (np.ndarray): Étiquettes de sortie de test de forme (n_samples,) ou (n_samples, n_classes) pour la classification multiclasse.
            curve_path (str): Chemin pour sauvegarder les courbes de performance.
            learning_rate (float): Taux d'apprentissage.
            nb_iter (int): Nombre d'itérations d'entraînement.
            partialsteps (int): Fréquence de sauvegarde du modèle et d'évaluation des métriques.
        """
        # On règle partialsteps lors du premier appel de train (puis on empéche de le changer pour avoir un bonne affichage de la courbe de performance)
        if self.partialsteps is None:
            self.partialsteps = partialsteps
        else:
            partialsteps = self.partialsteps

        if nb_iter == 1:
            self.progress_updated.emit(100)
        for i in range(nb_iter):
            A, Z= self.forward_propagation(X)

            if nb_iter > 1 :
                self.progress_updated.emit(int((i/(nb_iter-1))*100))

            if i % partialsteps == 0:
                self.L.append(self.loss(A, y))
                A_test, _ = self.forward_propagation(X_test)
                self.L_t.append(self.loss(A_test, y_test))
                y_pred_train = self.predict(X)
                y_pred_test = self.predict(X_test)
                self.evaluate_metrics(y, y_pred_train, y_test, y_pred_test)

            if i % (partialsteps*100) == 0:
                if self.path is not None:
                    self.save(self.path, curve_path ,bool_p=False)
            
            dW, db = self.back_propagation(Z, A, X, y)
            self.update(dW, db, learning_rate)  

    def get_model_type(self):
        """ Retourne le type de modèle : régression, binaire ou multiclass """
        if self.is_regression:
            return "regression"
        elif self.nb_classes == 1:
            return "binaire"
        else:
            return "multiclass"

    def save(self, filename, curve_path, bool_p=True):
        """ Sauvegarde du modèle dans un fichier 
        Args:
            filename (str): Chemin du fichier de sauvegarde.
            curve_path (str): Chemin pour sauvegarder les courbes de performance.
            bool_p (bool): Si True, affiche un message de confirmation."""
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
                'is_regression': self.is_regression,
                'partialsteps': self.partialsteps
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        courbe_perf(self, curve_path, bool_p)
        self.curve_save.emit()
        if bool_p:
            print(f"Modèle sauvegardé dans {filename}")

    def load(self, filename, bool_p=True):
        """ Charge un modèle depuis un fichier 
        Args:
            filename (str): Chemin du fichier de sauvegarde.
        """
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
            self.partialsteps = data['partialsteps'] if 'partialsteps' in data else None
        if bool_p:
            print(f"Modèle chargé depuis {filename}")
