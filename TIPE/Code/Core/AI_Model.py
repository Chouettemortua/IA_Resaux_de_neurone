
#  AI_Model.py

__init__ = "AI_Model"

# Importation de toute les bibliothèques nécessaires

import numpy as np
import shap
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
import pickle
from skopt import gp_minimize
from skopt.space import Real
from tqdm import tqdm 

# Pour éviter les problèmes d'affichage de matplotlib dans certains environnements
matplotlib.use('Agg')


#Classe pour les modèles de neurones (Neurone et plus utiliser c'est le prototype unineuronal)

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


# Pretraitement des données

def preprocecing(df, on, y_normalisation=True):
    """ Prétraite les données """

    def encodage(df):
        """ Encode les variables catégorielles """

        code_bmi = {'Normal':1,'Normal Weight': 1, 'Overweight': 3, 'Underweight': 4, 'Obesity': 5}
        code_gender = {'Male': 1, 'Female': 2}
        code_occupation = {'Software Engineer': 1, 'Doctor': 1, 'Sales Representative': 1, 'Nurse': 1, 'Teacher': 1,
                        'Scientist': 1, 'Engineer': 1, 'Lawyer': 1, 'Accountant': 1, 'Salesperson': 1, 'Manager': 1}
        code_sleep_disorder = {'Normal': 1, 'Sleep Apnea': 2, 'Insomnia': 3}
        

        df['Blood Pressure'] = df['Blood Pressure'].str.split('/').str[0].astype(int)
        df['Sleep Disorder'] = df['Sleep Disorder'].apply(lambda x: x if x in ['Sleep Apnea', 'Insomnia'] else 'Normal')
        
        # Mapper les colonnes catégorielles
        df['BMI Category'] = df['BMI Category'].map(code_bmi).fillna(-1).astype(int)
        df['Gender'] = df['Gender'].map(code_gender).fillna(-1).astype(int)
        df['Occupation'] = df['Occupation'].map(code_occupation).fillna(-1).astype(int)
        df['Sleep Disorder'] = df['Sleep Disorder'].map(code_sleep_disorder).fillna(-1).astype(int)

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

        max_values = {'Gender': 2, 'Age': 130, 'Occupation': 5, 'Sleep Duration': 24, 
                      'Physical Activity Level': 200, 'Stress Level': 10, 'BMI Category': 4, 
                      'Blood Pressure': 200, 'Heart Rate': 200, 'Daily Steps': 50000, 'Quality of Sleep': 10}
       
        df['Gender'] = df['Gender'].div(max_values['Gender'])
        df['Age'] = df['Age'].div(max_values['Age'])
        df['Occupation'] = df['Occupation'].div(max_values['Occupation'])
        df['Sleep Duration'] = df['Sleep Duration'].div(max_values['Sleep Duration'])
        df['Physical Activity Level'] = df['Physical Activity Level'].div(max_values['Physical Activity Level'])
        df['Stress Level'] = df['Stress Level'].div(max_values['Stress Level'])
        df['BMI Category'] = df['BMI Category'].div(max_values['BMI Category'])
        df['Blood Pressure'] = df['Blood Pressure'].div(max_values['Blood Pressure'])
        df['Heart Rate'] = df['Heart Rate'].div(max_values['Heart Rate'])
        df['Daily Steps'] = df['Daily Steps'].div(max_values['Daily Steps'])
        if 'Quality of Sleep' in df.columns:
            df['Quality of Sleep'] = df['Quality of Sleep'].div(max_values['Quality of Sleep'])

        return df
    
    def normalisation_y(y):
        if on[0] == 'Quality of Sleep':
            max_value = 10
        return y / max_value
    
    def intern(df):
        df= encodage(df)
        df = imputation(df)

        for col in on:
            X = df.drop(columns= on, axis=1)
        y = df[on[0]].values.reshape(-1, 1)

        # Normalize features
        X = normalisation(X)
        if y_normalisation:
            # Normalize target variable
            y = normalisation_y(y) 
        return X, y

    trainset, testset = split_data(df)
    X_train, y_train = intern(trainset)
    X_test, y_test = intern(testset)
    

    return X_train, y_train, X_test, y_test   

def preprocecing_user(df, on=None):
    """ Prétraite les données """

    def encodage(df):
        code_bmi = {'Normal': 1, 'Normal Weight': 1, 'Overweight': 2, 'Underweight': 3, 'Obese': 4}
        code_gender = {'Male': 1, 'Female': 2}
        code_occupation = {'working':1, 'unemployed':2, 'student':3, 'retired':4, 'other':5}

        df['Blood Pressure'] = df['Blood Pressure'].astype(str).str.split('/').str[0]
        df['Blood Pressure'] = pd.to_numeric(df['Blood Pressure'], errors='coerce')


        # Mapper les colonnes catégorielles
        #print("BMI uniques reçus :", df['BMI Category'].unique())
        df['BMI Category'] = df['BMI Category'].astype(str).str.strip().map(code_bmi)
        if df['BMI Category'].isnull().any():
            raise ValueError("Valeur invalide dans 'BMI Category'. Vérifiez vos entrées.")
        df['BMI Category'] = df['BMI Category'].astype(int)

        df['Gender']  = df['Gender'].astype(str).str.strip().map(code_gender)
        if df['Gender'].isnull().any():
            raise ValueError("Valeur invalide dans 'Gender'. Vérifiez vos entrées.")
        df['Gender'] = df['Gender'].astype(int)

        df['Occupation'] = df['Occupation'].astype(str).str.strip().map(code_occupation)
        if df['Occupation'].isnull().any():
            raise ValueError("Valeur invalide dans 'Occupation'. Vérifiez vos entrées.")
        df['Occupation'] = df['Occupation'].astype(int)
        

        return df

    def imputation(df):
        """ Impute les valeurs manquantes et supprime les colonnes inutiles """
        return df.fillna(df.mean())
    
    def normalisation(df):
        """ Normalise les données entre 0 et 1 """

        max_values = {'Gender': 2, 'Age': 130, 'Occupation': 5, 'Sleep Duration': 24, 
                      'Physical Activity Level': 200, 'Stress Level': 10, 'BMI Category': 4, 
                      'Blood Pressure': 200, 'Heart Rate': 200, 'Daily Steps': 50000}
       
        df['Gender'] = df['Gender'].div(max_values['Gender'])
        df['Age'] = df['Age'].div(max_values['Age'])
        df['Occupation'] = df['Occupation'].div(max_values['Occupation'])
        df['Sleep Duration'] = df['Sleep Duration'].div(max_values['Sleep Duration'])
        df['Physical Activity Level'] = df['Physical Activity Level'].div(max_values['Physical Activity Level'])
        df['Stress Level'] = df['Stress Level'].div(max_values['Stress Level'])
        df['BMI Category'] = df['BMI Category'].div(max_values['BMI Category'])
        df['Blood Pressure'] = df['Blood Pressure'].div(max_values['Blood Pressure'])
        df['Heart Rate'] = df['Heart Rate'].div(max_values['Heart Rate'])
        df['Daily Steps'] = df['Daily Steps'].div(max_values['Daily Steps'])

        return df
    
    def intern(df):
        if on is not None and on in df.columns:
            df.drop(columns=on, axis=1)
        else:
            df= encodage(df)
            df = imputation(df)
            df = normalisation(df) 
        return df

    return intern(df)

# Gestion des modèles

def load(path):
    """ Charge le dataset """
    return pd.read_csv(path)

def model_init(path_n, X_train, y_train, X_test, y_test, format, path, treshold_val=None, qt=None):
    """ Initialise le modèle """
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    model = Resaux(X_train, y_train, X_test, y_test, format, 1e-2, 1, path, threshold_val=treshold_val, qt=qt)
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

# Outils d'analyse

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

def affichage_perf(X_train, y_train, X_test, y_test, model, qt=None):
    """ Affiche automatiquement les métriques selon le type du modèle (régression, binaire, ou multi-classes) """

    y_train_true = y_train.flatten()
    y_test_true = y_test.flatten()

    # Prédictions
    pred_train = model.predict(X_train).flatten()
    pred_test = model.predict(X_test).flatten()

    # === CAS RÉGRESSION ===
    if model.nb_classes == 1 and qt is not None:
        y_train_true = qt.inverse_transform(y_train_true.reshape(-1, 1)).flatten()
        y_test_true  = qt.inverse_transform(y_test_true.reshape(-1, 1)).flatten()
        print("=== Régression - Train ===")
        print(f"MSE: {mean_squared_error(y_train_true, pred_train):.4f}")
        print(f"MAE: {mean_absolute_error(y_train_true, pred_train):.4f}")
        print(f"R²:  {r2_score(y_train_true, pred_train):.4f}\n")

        print("=== Régression - Test ===")
        print(f"MSE: {mean_squared_error(y_test_true, pred_test):.4f}")
        print(f"MAE: {mean_absolute_error(y_test_true, pred_test):.4f}")
        print(f"R²:  {r2_score(y_test_true, pred_test):.4f}")

    # === CAS CLASSIFICATION BINAIRE ===
    elif model.nb_classes == 1:
        y_train_bin = (y_train_true >= model.treshold_val).astype(int)
        y_test_bin  = (y_test_true >= model.treshold_val).astype(int)

        print("=== Classification binaire ===")
        print(f"Train Accuracy:  {accuracy_score(y_train_bin, pred_train):.4f}")
        print(f"Test Accuracy:   {accuracy_score(y_test_bin, pred_test):.4f}")
        print(f"Train F1 Score:  {f1_score(y_train_bin, pred_train):.4f}")
        print(f"Test F1 Score:   {f1_score(y_test_bin, pred_test):.4f}")
        print(f"Train Precision: {precision_score(y_train_bin, pred_train):.4f}")
        print(f"Test Precision:  {precision_score(y_test_bin, pred_test):.4f}")
        print(f"Train Recall:    {recall_score(y_train_bin, pred_train):.4f}")
        print(f"Test Recall:     {recall_score(y_test_bin, pred_test):.4f}")

    # === CAS CLASSIFICATION MULTI-CLASSES ===
    else:
        y_train_int = y_train_true.astype(int)
        y_test_int  = y_test_true.astype(int)
        pred_train = np.round(pred_train).astype(int)
        pred_test = np.round(pred_test).astype(int)

        print("=== Classification multi-classes ===")
        print(f"Train Accuracy:  {accuracy_score(y_train_int, pred_train):.4f}")
        print(f"Test Accuracy:   {accuracy_score(y_test_int, pred_test):.4f}")
        print(f"Train F1 Score:  {f1_score(y_train_int, pred_train, average='weighted'):.4f}")
        print(f"Test F1 Score:   {f1_score(y_test_int, pred_test, average='weighted'):.4f}")
        print(f"Train Precision: {precision_score(y_train_int, pred_train, average='weighted', zero_division=0):.4f}")
        print(f"Test Precision:  {precision_score(y_test_int, pred_test, average='weighted', zero_division=0):.4f}")
        print(f"Train Recall:    {recall_score(y_train_int, pred_train, average='weighted', zero_division=0):.4f}")
        print(f"Test Recall:     {recall_score(y_test_int, pred_test, average='weighted', zero_division=0):.4f}")

def courbe_perf(sleep, path, bool_p=True):
    """ Met les courbes de perte et de performance dans un fichier """
    plt.figure(figsize=(12, 4))

    # Titre et label dynamique selon le type de modèle
    if sleep.is_regression:
        acc_label = "R²"
        acc_title = "Courbe de R²"
        acc_ylabel = "R² score"
    else:
        acc_label = "accuracy"
        acc_title = "Courbe d'accuracy"
        acc_ylabel = "Accuracy"

    # Perte (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(sleep.L, label="train loss")
    plt.plot(sleep.L_t, label="test loss")
    plt.legend()
    plt.title("Courbe de perte")
    plt.xlabel("Itérations")
    plt.ylabel("Loss")

    # Performance (Accuracy ou R²)
    plt.subplot(1, 2, 2)
    plt.plot(sleep.acc, label=f"train {acc_label}")
    plt.plot(sleep.acc_t, label=f"test {acc_label}")
    plt.legend()
    plt.title(acc_title)
    plt.xlabel("Itérations")
    plt.ylabel(acc_ylabel)

    plt.tight_layout()
    plt.savefig(path)

    if bool_p:
        print("Courbes sauvegardées dans", path)

def val_evolution(model, input_row, modifiable_features, modifiable_indices, features, nb_iter=30):
    # Dictionnaire des max utilisés pour la normalisation
    max_values = {
        'Sleep Duration': 24, 
        'Physical Activity Level': 200, 
        'Stress Level': 10, 
        'BMI Category': 4, 
        'Blood Pressure': 200, 
        'Heart Rate': 200, 
        'Daily Steps': 50000
    }

    # Définir les bornes réalistes (normalisées)
    bounds = {
        'Sleep Duration': (6 / 24, 9 / 24),  # 6 à 9 heures de sommeil
        'Physical Activity Level': (20 / 200, 150 / 200),  # 20 à 150 min activité
        'Stress Level': (0 / 10, 7 / 10),  # idéalement on cherche à le réduire
        'BMI Category': (1 / 4, 2 / 4),  # viser "Normal" (1) ou "Overweight" (2)
        'Daily Steps': (1000 / 50000, 30000 / 50000),  # cible santé classique
    }

    # Construction des bornes pour gp_minimize
    dimensions = [Real(*bounds[feat]) for feat in modifiable_features]

    input_row = input_row.flatten()

    def objective(x):
        modified_input = input_row.copy()
        for i, idx in enumerate(modifiable_indices):
            modified_input[idx] = np.clip(x[i], 0, 1)
        prediction = model.predict(modified_input.reshape(1, -1))[0]
        return -prediction

    res = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=nb_iter,
        random_state=123
    )

    # Affichage du résultat
    print("Amélioration maximale prédite : %.4f" % (-res.fun))
    print("Nouvelles valeurs modifiées (dénormalisées) :")
    for i, idx in enumerate(modifiable_indices):
        feat = features[idx]
        val = res.x[i] * max_values[feat]
        if feat == "BMI Category":
            val = round(val)  # Catégorie entière
        elif feat == "Stress Level":
            val = round(val, 1)  # un chiffre après la virgule suffit
        elif feat == "Daily Steps":
            val = int(val)
        else:
            val = round(val, 2)
        print(f"  {feat}: {val}")

# Fonctions principales

def main_quality_of_sleep(bool_c, bool_t, path_n, path_c, verbose=False):
    """ Main function for Resaux on the quality of sleep dataset """
    
    # Load the dataset

    data = load('TIPE/Code/Data/Sleep_health_and_lifestyle_dataset.csv')
    df = data.copy()

    # Define ami

    ami = [0,0.3,0.2,0.47,0.11,4,0,0.88,0.89,0.5]

    #Preprocessing

    # Uncomment the following line to see the dataset before preprocessing
    # analyse_pre_process(df)
    
    X_train, y_train, X_test, y_test = preprocecing(df, ['Quality of Sleep', 'Sleep Disorder'], y_normalisation=True)

    assert not np.any(np.isin(X_train.index, X_test.index))

    # Uncomment the following line to see the dataset after preprocessing
    # analyse_post_process(X_train, y_train, X_test, y_test)

    # Transformer y_train / y_test avec QuantileTransformer
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    y_train = qt.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = qt.transform(y_test.reshape(-1, 1)).flatten()
    
    # Train the model
    if bool_c:
        sleep = model_init(path_n, X_train, y_train, X_test, y_test, [128,64,32,16,1], path_n, treshold_val=0.5, qt=qt)
    else: 
        sleep = model_charge(path_n)
    if bool_t:
        sleep = model_train(X_train, y_train, X_test, y_test, sleep, path_n)

    # Ami évaluation affichage

    ami_in = np.array(ami).reshape(1,-1)
    ami_pred = sleep.predict(ami_in)[-1].flatten()
    print(f"mons amis: {ami_pred}") 

    # affichage des performances

    affichage_perf(X_train, y_train, X_test, y_test, sleep, qt)    

    # evolution des variables modifiables pour améliorer la prédiction
    features = ["Gender", "Age", "Occupation", "Sleep Duration",
                "Physical Activity Level", "Stress Level", "BMI Category",
                "Blood Pressure", "Heart Rate", "Daily Steps"]
    
    non_modifiables = ["Quality of Sleep", "Sleep Disorder", 'Age', 'Occupation', 'Gender', 'Heart Rate', 'Blood Pressure']

    modifiable_indices = [i for i, f in enumerate(features) if f not in non_modifiables]
    modifiable_features = [features[i] for i in modifiable_indices]

    if verbose:
        print("")
        print("Evolution des variables modifiables pour améliorer la prédiction :")
        val_evolution(sleep, ami_in, modifiable_features, modifiable_indices, features, nb_iter=30)
        print(ami_in.shape)

        # explainer
        explainer = shap.KernelExplainer(sleep.predict, X_train)

        # Calculer les valeurs SHAP
        shap_values = explainer.shap_values(ami_in, nsamples=100)

        # Visualiser
        shap.initjs()
        shap.force_plot(explainer.expected_value, shap_values, ami_in)
        shap.summary_plot(shap_values, features=features, feature_names=features, show=False)
        plt.savefig('TIPE/Code/Saves_Curve/values', bbox_inches='tight')
        plt.close()


    #courbe_perf(sleep)
    if bool_c or bool_t:     
        courbe_perf(sleep, path_c)

def main_sleep_trouble(boul_c, bool_t, path_n, path_c):
    """ Main function for Resaux on the sleep trouble dataset """
    
    # Load the dataset
    data = load('TIPE/Code/Data/Sleep_health_and_lifestyle_dataset.csv')
    df = data.copy()

    # Preprocessing
    # Uncomment the following line to see the dataset before preprocessing
    # analyse_pre_process(df)
    
    X_train, y_train, X_test, y_test = preprocecing(df, ['Sleep Disorder'], y_normalisation=False)

    assert not np.any(np.isin(X_train.index, X_test.index))

    # Uncomment the following line to see the dataset after preprocessing
    # analyse_post_process(X_train, y_train, X_test, y_test)
    
    # Train the model

    # architecture = [1,30,75,500,1000,500,100,75,30,1] atteint les 0.8 
    # architecture = [1,2000,1500,1000,500,400,100,75,30,1] atteint les 0.85 mais pas stable et long -> set trop petit pour le nombre de neurones
    # architecture = [1, 64, 32, 16, 1] se stabilise à 0.5 sous apprentissage
    
    if boul_c:
        sleep = model_init(path_n, X_train, y_train, X_test, y_test, [256, 128, 64, 32, 16, 4], path_n, treshold_val=None)
    else: 
        sleep = model_charge(path_n)

    if bool_t:
        sleep = model_train(X_train, y_train, X_test, y_test, sleep, path_n, iteration=1000, precision=1e-2)

    # Ami évaluation affichage
    ami = [0,0.3,0.6,0.47,0.8,0.11,4,0,0.88,0.89,0.5]
    print(f"ami: {(sleep.predict(np.array(ami).reshape(1,-1)))}")

    # affichage des performances

    affichage_perf(X_train, y_train, X_test, y_test, sleep)

    # courbe_perf(sleep)
    if boul_c or bool_t:     
        courbe_perf(sleep,path_c)

# Lancement automatique des fonctions principales
if __name__ == "__main__":
    # Main function launcher with arguments
    main_quality_of_sleep(False, False, "TIPE/Code/Saves/save_sleep_quality.pkl", "TIPE/Code/Saves_Curves/curve_sleep_quality.png")
    main_sleep_trouble(False, False, "TIPE/Code/Saves/save_sleep_trouble.pkl", "TIPE/Code/Saves_Curves/curve_sleep_trouble.png")
  