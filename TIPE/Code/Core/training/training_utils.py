
__init__ = "training_utils"

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score
from skopt import gp_minimize
from skopt.space import Real


from ..models.AI_Model import Resaux

# Gestion des modèles

def load(path):
    """ Charge le dataset """
    return pd.read_csv(path)

def model_init(path_n, X_train, y_train, X_test, y_test, format, path, treshold_val=None, qt=None):
    """ Initialise le modèle """
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    model = Resaux(X_train, y_train, X_test, y_test, format, path, threshold_val=treshold_val, qt=qt)
    model.save(path_n)
    return model  

def model_train(X_train, y_train, X_test, y_test, model, path_n, iteration=1000, learning_rate =1e-2):
    """ Entraîne le modèle """

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    model.train(X_train, y_train, X_test, y_test, learning_rate, iteration)
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
        print(f"R²:  {r2_score(y_train_true, pred_train):.4f}")

        print("\n=== Régression - Test ===")
        print(f"MSE: {mean_squared_error(y_test_true, pred_test):.4f}")
        print(f"MAE: {mean_absolute_error(y_test_true, pred_test):.4f}")
        print(f"R²:  {r2_score(y_test_true, pred_test):.4f}")

    # === CAS CLASSIFICATION BINAIRE ===
    elif model.nb_classes == 1:
        y_train_bin = (y_train_true >= model.treshold_val).astype(int)
        y_test_bin  = (y_test_true >= model.treshold_val).astype(int)

        print("=== Classification binaire - Train ===")
        print(f"Accuracy:  {accuracy_score(y_train_bin, pred_train):.4f}")  
        print(f"F1 Score:  {f1_score(y_train_bin, pred_train):.4f}")
        print(f"Precision: {precision_score(y_train_bin, pred_train):.4f}") 
        print(f"Recall:    {recall_score(y_train_bin, pred_train):.4f}")
        
        print("\n=== Classification binaire - Test ===")
        print(f"Accuracy:   {accuracy_score(y_test_bin, pred_test):.4f}")
        print(f"F1 Score:   {f1_score(y_test_bin, pred_test):.4f}")
        print(f"Precision:  {precision_score(y_test_bin, pred_test):.4f}")
        print(f"Recall:     {recall_score(y_test_bin, pred_test):.4f}")

    # === CAS CLASSIFICATION MULTI-CLASSES ===
    else:
        y_train_int = y_train_true.astype(int)
        y_test_int  = y_test_true.astype(int)
        pred_train = np.round(pred_train).astype(int)
        pred_test = np.round(pred_test).astype(int)

        print("=== Classification multi-classes - Train ===")
        print(f"Accuracy:  {accuracy_score(y_train_int, pred_train):.4f}")
        print(f"F1 Score:  {f1_score(y_train_int, pred_train, average='weighted'):.4f}")
        print(f"Precision: {precision_score(y_train_int, pred_train, average='weighted', zero_division=0):.4f}") 
        print(f"Recall:    {recall_score(y_train_int, pred_train, average='weighted', zero_division=0):.4f}")
        
        print("\n=== Classification multi-classes - Test ===")
        print(f"Accuracy:   {accuracy_score(y_test_int, pred_test):.4f}")
        print(f"F1 Score:   {f1_score(y_test_int, pred_test, average='weighted'):.4f}")
        print(f"Precision:  {precision_score(y_test_int, pred_test, average='weighted', zero_division=0):.4f}")
        print(f"Recall:     {recall_score(y_test_int, pred_test, average='weighted', zero_division=0):.4f}")

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
