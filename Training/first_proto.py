import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pickle
from tqdm import tqdm 


class Neurone: 

    # Sert a l'initialisation du vecteur des paramétre sur les entrée
    # et la constante
    def __init__(self, X=None, y=None, X_test=None, y_test=None, learning_rate=0.1, nb_iter=100):

        #Paramètre definisant mon neurone
        self.W = None
        self.b = None
        self.L = []
        self.L_t = []
        self.acc = []
        self.acc_t = []

        #si on fournit des paramètre on fait automatiquement un entrainement dessus
        if X is not None and y is not None:
            self.W = np.random.randn(X.shape[1], 1)
            self.b = np.random.randn(1)
            self.train(X, y, X_test, y_test, learning_rate, nb_iter)


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


    def train(self, X, y, X_test, y_test, learning_rate=0.1, nb_iter=100):
        #boucle d'update
        for i in tqdm(range(nb_iter)):
            #activation
            A = self.model(X)

            if i%10==0 :
                #Calcule coût
                self.L.append(self.log_loss(A, y))
                self.L_t.append(self.log_loss(self.model(X_test), y_test))

                #Calcul accuracy
                pred = self.predict(X)
                pred_t = self.predict(X_test)
                self.acc.append(accuracy_score(y,pred))
                self.acc_t.append(accuracy_score(y_test, pred_t))

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
            pickle.dump({'W': self.W, 'b': self.b, 'L': self.L, 'L_t': self.L_t, 'acc':self.acc, 'acc_t':self.acc_t}, f)
        print(f"Modèle sauvegardé dans {filename}")

    # Charger les paramètres (poids, biais et historique de perte, historique accuracy)
    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.W = data['W']
            self.b = data['b']
            self.L = data['L']
            self.L_t = data['L_t']
            self.acc = data['acc']
            self.acc_t = data['acc_t']
        print(f"Modèle chargé depuis {filename}")


def main(bool_c, bool_t, path, path_c):

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


    #Nouveau neurone
    if bool_c == 0:
        chien_chat = Neurone(X_train_r, y_train, X_test_r, y_test, 1e-2)
        chien_chat.save(path)
        
    #Neurone existant
    else:
            chien_chat = Neurone()
            chien_chat.load(path)

    #celon si on veut entrainer ou non
    if bool_t == 1:
        chien_chat.train(X_train_r, y_train, X_test_r, y_test, 1e-2, 10000)
        chien_chat.save(path)


    #affichage et calcule d'efficacité sur mes test
    pred_chien_chat = chien_chat.predict(X_train_r)
    print(accuracy_score(y_train, pred_chien_chat))
    pred_chien_chat_test = chien_chat.predict(X_test_r)
    print(accuracy_score(y_test, pred_chien_chat_test))


    #création et stockage des courbesclear
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(chien_chat.L, label="train loss")
    plt.plot(chien_chat.L_t, label="test loss")
    plt.legend()
    plt.title("Courbe de perte")
    plt.xlabel("Itérations")
    plt.ylabel("Loss")

    plt.subplot(1,2,2)
    plt.plot(chien_chat.acc, label="train acc")
    plt.plot(chien_chat.acc_t, label="test acc")
    plt.legend()
    plt.title("Courbe d'accuracy")
    plt.xlabel("Itérations")
    plt.ylabel("Acc")

    plt.savefig(path_c)  # Sauvegarde dans un fichier
    print("Courbes sauvegardée dans ", path_c)


main(0, 1, "Training/saves/save_chien_chat.pkl", "Training/saves/curve.png")