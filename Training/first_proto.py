import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import pickle


class Neurone: 

    # Sert a l'initialisation du vecteur des paramétre sur les entrée
    # et la constante
    def __init__(self, X=None, y=None, learning_rate=0.1, nb_iter=100):

        #Paramètre definisant mon neurone
        self.W = None
        self.b = None
        self.L = []
        self.acc = []

        #si on fournit des paramètre on fait automatiquement un entrainement dessus
        if X is not None and y is not None:
            self.W = np.random.randn(X.shape[1], 1)
            self.b = np.random.randn(1)
            self.train(X, y, learning_rate, nb_iter)


    # definit le modèle donc la fonction linéaire
    # puis transforme la sortie linéaire en un pourcentage
    def model(self,X):
        Z = X.dot(self.W)+self.b
        A = 1/(1+np.exp(-Z)) 
        return A


    #On definit la fonction de coût donc log loss ici
    def log_loss(self,A,y):
        epsilon = 1e-15
        return 1/len(y) * np.sum(-y*np.log(A+epsilon) - (1-y)*np.log(1-A+epsilon))


    #On definit nos gradient
    def gradients(self,A,X,y):
        dW = 1/len(y) * np.dot(X.T,A-y)
        db = 1/len(y) * np.sum(A-y)
        return dW,db


    #On definit la decente de gradient pour ajuster les paramètre
    def update(self,dW, db, learning_rate):
        self.W = self.W-learning_rate*dW
        self.b = self.b-learning_rate*db


    def train(self,X, y,learning_rate=0.1, nb_iter=100):
        #boucle d'update
        for _ in range(nb_iter):
            #activation
            A = self.model(X)

            #Calcule coût
            self.L.append(self.log_loss(A, y))

            #Calcul accuracy
            pred = self.predict(X)
            self.acc.append(accuracy_score(y,pred))

            #Mise a jour
            dW,db = self.gradients(A,X,y)
            self.update(dW, db, learning_rate)
    

    #Calcul des prediction pour un set de donné
    def predict(self, X):
        A = self.model(X)
        return A >= 0.5


    #sauvegarde les parra dans un fichier
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'W': self.W, 'b': self.b, 'L': self.L, 'acc':self.acc}, f)
        print(f"Modèle sauvegardé dans {filename}")

    # Charger les paramètres (poids, biais et historique de perte, historique accuracy)
    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.W = data['W']
            self.b = data['b']
            self.L = data['L']
            self.acc = data['acc']
        print(f"Modèle chargé depuis {filename}")


def main(bool_c, bool_t):

    def load_data():
        train_dataset = h5py.File('Training/datasets/trainset.hdf5', "r")
        X_train = np.array(train_dataset["X_train"][:]) # your train set features
        y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

        test_dataset = h5py.File('Training/datasets/testset.hdf5', "r")
        X_test = np.array(test_dataset["X_test"][:]) # your test set features
        y_test = np.array(test_dataset["Y_test"][:]) # your test set labels
        
        return X_train, y_train, X_test, y_test

    X_train, y_train, X_test, y_test = load_data()

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    # Reshape des labels
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    """
    # Vérification des dimensions
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    #dimension obtenu
    X_train shape: (1000, 4096)
    y_train shape: (1000, 1)
    X_test shape: (200, 4096)
    y_test shape: (200, 1)
    """


    X_train_r=X_train/X_train.max()
    X_test_r=X_test/X_train.max()


    path = "Training/saves/save_chien_chat.pkl"

    #Nouveau neurone
    if bool_c == 0:
        chien_chat = Neurone(X_train_r, y_train, 1e-3)
        chien_chat.save(path)
        
    #Neurone existant
    else:
            chien_chat = Neurone()
            chien_chat.load(path)

    #celon si on veut entrainer ou non
    if bool_t == 1:
        chien_chat.train(X_train_r, y_train, 1e-3, 1000)
        chien_chat.save(path)


    #affichage et calcule d'efficacité sur mes test
    pred_chien_chat = chien_chat.predict(X_train_r)
    print(accuracy_score(y_train, pred_chien_chat))
    pred_chien_chat_test = chien_chat.predict(X_test_r)
    print(accuracy_score(y_test, pred_chien_chat_test))


    #création et stockage des courbes
    plt.plot(chien_chat.L)
    plt.title("Courbe de perte")
    plt.xlabel("Itérations")
    plt.ylabel("Loss")
    plt.savefig("Training/saves/loss_curve.png")  # Sauvegarde dans un fichier
    print("Courbe de perte sauvegardée dans Training/saves/loss_curve.png")

    plt.plot(chien_chat.acc)
    plt.title("Courbe d'accuracy")
    plt.xlabel("Itérations")
    plt.ylabel("Acc")
    plt.savefig("Training/saves/acc_curve.png")  # Sauvegarde dans un fichier
    print("Courbe d'accuracy sauvegardée dans Training/saves/acc_curve.png")


main(0, 1)