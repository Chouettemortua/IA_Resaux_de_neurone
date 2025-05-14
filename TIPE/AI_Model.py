import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm 

matplotlib.use('Agg')


#Classe pour les modèles de neurones

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

class Resaux:
    def __init__(self, X=None, y=None, X_test=None, y_test=None, nb_neurone_couche=[1], learning_rate=0.1, nb_iter=1, path=None):
        """ Initialise le réseau de neurones """
        nb_neurone_couche.reverse()
        self.path = path
        self.W = None
        self.b = None
        self.nb_neurone_couche = nb_neurone_couche 
        self.L = []
        self.L_t = []  
        self.acc = []
        self.acc_t = []

        if X is not None and y is not None:
            self.W = [np.random.randn(nb_neurone_couche[0], X.shape[1])]
            self.W = self.W+[np.random.randn(nb_neurone_couche[i], nb_neurone_couche[i-1]) for i in range(1, len(nb_neurone_couche))]
            self.b = [np.random.randn(n, 1  ) for n in nb_neurone_couche]
            self.train(X, y, X_test, y_test, learning_rate, nb_iter)

    def forward_propagation(self, X):
        """ Calcule la propagation avant du réseau de neurones """
        Z = [np.dot(self.W[0], X.T) + self.b[0]]
        A = [1 / (1 + np.exp(-Z[0]))]
        for i in range(1, len(self.W)):
            #print(f"W[{i}].shape", self.W[i].shape)
            #print(f"A[{i-1}].shape", A[i-1].shape)
            Z.append(self.W[i].dot(A[i-1]) + self.b[i])
            A.append(1 / (1+np.exp(-Z[i])))
        return A
    
    def back_propagation(self, A, X, y):
        """ Calcule les gradients de la fonction de perte par rapport aux poids et au biais """
        m = X.shape[0]
        y = y.reshape(1, -1)  # y doit être (1, m)

        dZ = A[-1] - y  # (1, m)
        dW = []
        db = []

        for i in reversed(range(len(self.W))):
            A_prev = A[i - 1] if i > 0 else X.T  # (n_l-1, m)

            dW_i = np.dot(dZ, A_prev.T) / m
            db_i = np.sum(dZ, axis=1, keepdims=True) / m

            dW.insert(0, dW_i)
            db.insert(0, db_i)

            if i > 0:
                dA_prev = np.dot(self.W[i].T, dZ)
                dZ = dA_prev * A[i - 1] * (1 - A[i - 1])  # sigmoid prime

        return dW, db

    def log_loss(self, A, y):
        """ Calcule la fonction de perte logistique """
        epsilon = 1e-15
        A = np.clip(A[-1], epsilon, 1 - epsilon)
        return -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))

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
        """ Prédiction sur de nouvelles données """
        A = self.forward_propagation(X)
        return (A[-1] >= 0.5).astype(int).flatten()

    def train(self, X, y, X_test, y_test, learning_rate=1e-2, nb_iter=10000, partialsteps=10):
        """ Entraîne le modèle sur les données d'entraînement """
        for i in tqdm(range(nb_iter)):
            A = self.forward_propagation(X)

            if i % partialsteps == 0:
                self.L.append(self.log_loss(A, y))
                self.L_t.append(self.log_loss(self.forward_propagation(X_test), y_test))
                self.acc.append(accuracy_score(y >= 0.5, self.predict(X).flatten()))
                self.acc_t.append(accuracy_score(y_test >= 0.5, self.predict(X_test)))
            if i % (partialsteps*100) == 0:
                if self.path is not None:
                    self.save(self.path, bool_p=False)
            
            dW, db = self.back_propagation(A, X, y)
            self.update(dW, db, learning_rate)  

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
                'path': self.path
            }, f)
        courbe_perf(self, self.path.replace(".pkl", ".png").replace("save", "curve"), bool_p)
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
        print(f"Modèle chargé depuis {filename}")


#Fonctions de prétraitement et d'analyse et pour le modèle

def load(path):
    """ Charge le dataset """
    return pd.read_csv(path)

def preprocecing(df, on):
    """ Prétraite les données """

    def encodage(df):
        """ Encode les variables catégorielles """

        code = {'Normal': 0, 'Normal Weight':0, 'Overweight': 1, 'Underweight':2, 'Obesity': 3, 'Software Eginneer': 0, 'Doctor': 1, 'Sales Representative': 2, 
                'Nurse': 3, 'Teacher': 4, 'Scientist': 5, 'Engineer': 6, 'Lawyer': 7, 'Accountant': 8, 'Salesperson': 9, 'Manager': 10,
                'Sleep Apnea': 1, 'Insomnia': 2, 'Male': 0, 'Female': 1 }
        

        df['Blood Pressure'] = df['Blood Pressure'].str.split('/').str[0].astype(int)
        df['Sleep Disorder'] = df['Sleep Disorder'].apply(lambda x: x if x in ['Sleep Apnea', 'Insomnia'] else 'Normal')
        
        for col in df.select_dtypes('object'):
            df[col] = df[col].map(code)

        return df

    def imputation(df):
        """ Impute les valeurs manquantes et supprime les colonnes inutiles """
        df = df.drop(columns=['Person ID'])
        return df.fillna(df.mean())
    
    def split_data(df):
        """ Sépare le dataset en train et test """

        trainset, testset = train_test_split(df, test_size=0.2, random_state=42)
        return trainset, testset
    
    def normalisation(df):
        """ Normalise les données entre 0 et 1 """

        return df/df.max()
    
    def intern(df):
        df= encodage(df)
        df = imputation(df)

        X = df.drop(columns= on, axis=1)
        y = df[on].values.reshape(-1, 1)

        # Normalize features
        X = normalisation(X)
        y = normalisation(y) 

        return X, y

    trainset, testset = split_data(df)
    X_train, y_train = intern(trainset)
    X_test, y_test = intern(testset)
    

    return X_train, y_train, X_test, y_test   

def model_init(path_n, X_train, y_train, X_test, y_test, format, path):
    """ Initialise le modèle """
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    model = Resaux(X_train, y_train, X_test, y_test, format, 1e-2, 1, path)
    model.save(path_n)
    return model  

def model_train(X_train, y_train, X_test, y_test, model, path_n, iteration=1000, precision =1e-2):
    """ Entraîne le modèle """

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    model.train(X_train, y_train, X_test, y_test, precision, iteration)
    model.save(path_n)
    return model

def model_charge(path_n):
    model = Resaux()
    model.load(path_n)
    return model

def analyse_pre_process(df):
    ''' Analyse le dataset avant le prétraitement '''
    print()
    print(df.head())
    print()
    print(df.info())
    print()
    print(df.describe())
    print()
    print(df.isna().sum()/df.shape[0])
    print()

def analyse_post_process(X_train, y_train, X_test, y_test):
    ''' Analyse le dataset après le prétraitement '''
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("\nX_train data types:\n", pd.DataFrame(X_train).dtypes)
    print("\nFirst 5 rows of X_train:\n", pd.DataFrame(X_train).head())
    print("\ny_train data types:\n", pd.DataFrame(y_train).dtypes)
    print("\nFirst 5 rows of y_train:\n", pd.DataFrame(y_train).head())

def affichage_perf(X_train, y_train, X_test, y_test, model):
    """ Affiche les performances du modèle """
    y_pred_train = model.predict(X_train).flatten()
    y_pred_test = model.predict(X_test).flatten()

    initial_pred_train = model.acc[0]
    initial_pred_test = model.acc_t[0]
    print("Initial Train Accuracy:", initial_pred_train)
    print("Initial Test Accuracy:", initial_pred_test)


    print("Train Accuracy:", accuracy_score(y_train >= 0.5, y_pred_train))
    print("Test Accuracy:", accuracy_score(y_test >= 0.5, y_pred_test))

    # Additional metrics
    print("Train F1 Score:", f1_score(y_train >= 0.5, y_pred_train, average='weighted', zero_division= np.nan))
    print("Test F1 Score:", f1_score(y_test >= 0.5, y_pred_test, average='weighted', zero_division= np.nan))
    print("Train Precision:", precision_score(y_train >= 0.5, y_pred_train, average='weighted', zero_division= np.nan))
    print("Test Precision:", precision_score(y_test >= 0.5, y_pred_test, average='weighted', zero_division= np.nan))
    print("Train Recall:", recall_score(y_train >= 0.5, y_pred_train, average='weighted', zero_division= np.nan))
    print("Test Recall:", recall_score(y_test >= 0.5, y_pred_test, average='weighted', zero_division= np.nan))

def courbe_perf(sleep, path, bool_p=True):
    """ Met les courbes de perte et d'accuracy dans un fichier """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(sleep.L, label="train loss")
    plt.plot(sleep.L_t, label="test loss")
    plt.legend()
    plt.title("Courbe de perte")
    plt.xlabel("Itérations")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(sleep.acc, label="train acc")
    plt.plot(sleep.acc_t, label="test acc")
    plt.legend()
    plt.title("Courbe d'accuracy")
    plt.xlabel("Itérations")
    plt.ylabel("Acc")

    plt.savefig(path)
    if bool_p:
        print("Courbes sauvegardée dans ", path)


# Fonctions principales

def main_quality_of_sleep(bool_c, bool_t, path_n, path_c):
    """ Main function for Resaux on the quality of sleep dataset """
    
    # Load the dataset

    data = load('TIPE/datasets/Sleep_health_and_lifestyle_dataset.csv')
    df = data.copy()

    # Define ami

    ami = [0,0.3,0.6,0.47,0.11,4,0,0.88,0.89,0.5,0]

    #Preprocessing

    # Uncomment the following line to see the dataset before preprocessing
    # analyse_pre_process(df)
    
    X_train, y_train, X_test, y_test = preprocecing(df, 'Quality of Sleep')

    assert not np.any(np.isin(X_train.index, X_test.index))

    # Uncomment the following line to see the dataset after preprocessing
    # analyse_post_process(X_train, y_train, X_test, y_test)
    
    # Train the model
    if bool_c:
        sleep = model_init(path_n, X_train, y_train, X_test, y_test, [1,3,3,1], path_n)
    else: 
        sleep = model_charge(path_n)

    if bool_t:
        sleep = model_train(X_train, y_train, X_test, y_test, sleep, path_n)

    # Ami évaluation affichage

    print(f"mons amis :{sleep.predict(np.array(ami))}") 

    # affichage des performances

    affichage_perf(X_train, y_train, X_test, y_test, sleep)

    #courbe_perf(sleep)
    if bool_c or bool_t:     
        courbe_perf(sleep, path_c)

def main_sleep_trouble(boul_c, bool_t, path_n, path_c):
    """ Main function for Resaux on the sleep trouble dataset """
    
    # Load the dataset
    data = load('TIPE/datasets/Sleep_health_and_lifestyle_dataset.csv')
    df = data.copy()

    # Preprocessing
    # Uncomment the following line to see the dataset before preprocessing
    # analyse_pre_process(df)
    
    X_train, y_train, X_test, y_test = preprocecing(df, 'Sleep Disorder')

    assert not np.any(np.isin(X_train.index, X_test.index))

    # Uncomment the following line to see the dataset after preprocessing
    # analyse_post_process(X_train, y_train, X_test, y_test)
    
    # Train the model

    # architecture = [1,30,75,500,1000,500,100,75,30,1] atteint les 0.8 
    # architecture = [1,2000,1500,1000,500,400,100,75,30,1] atteint les 0.85 mais pas stable et long -> set trop petit pour le nombre de neurones
    # architecture = [1, 64, 32, 16, 1] se stabilise à 0.5 sous apprentissage

    if boul_c:
        sleep = model_init(path_n, X_train, y_train, X_test, y_test, [1, 256, 128, 64, 32, 1], path_n)
    else: 
        sleep = model_charge(path_n)

    if bool_t:
        sleep = model_train(X_train, y_train, X_test, y_test, sleep, path_n, iteration=5000, precision=1e-2)

    # affichage des performances

    affichage_perf(X_train, y_train, X_test, y_test, sleep)

    # courbe_perf(sleep)
    if boul_c or bool_t:     
        courbe_perf(sleep,path_c)

if __name__ == "__main__":
    # Main function launcher with arguments
    main_quality_of_sleep(False, False, "TIPE/Saves/save_sleep_quality.pkl", "TIPE/Saves/curve_sleep_quality.png")
    main_sleep_trouble(False, False, "TIPE/Saves/save_sleep_trouble.pkl", "TIPE/Saves/curve_sleep_trouble.png")
  