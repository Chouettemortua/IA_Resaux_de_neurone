
__init__ = "training_utils"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, mean_squared_error, mean_absolute_error, 
    r2_score, confusion_matrix, ConfusionMatrixDisplay)
from skopt import gp_minimize
from skopt.space import Real
import os


from ..models.AI_Model import Resaux

# Gestion des modèles

def load(path):
    """ Charge le dataset 
    args:
        path (str): chemin du fichier CSV"""
    return pd.read_csv(path)

def model_init(path_n, path_c, X_train, y_train, X_test, y_test, format, treshold_val=None, qt=None, verbose=False):
    """ Initialise le modèle 
    args:
        path_n (str): chemin pour sauvegarder le modèle
        X_train (np.array): données d'entrainement
        y_train (np.array): labels d'entrainement
        X_test (np.array): données de test
        y_test (np.array): labels de test
        format (str): type du modèle ('regression', 'binaire', 'multi-classes')
        treshold_val (float): seuil pour la classification binaire
        qt (QuantileTransformer): transformateur pour la régression
        returns:
            model (Resaux): modèle initialisé"""
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    model = Resaux(X_train, y_train, X_test, y_test, format, path_n, threshold_val=treshold_val, qt=qt)
    model.save(path_n, path_c, verbose)
    return model  

def model_train(X_train, y_train, X_test, y_test, model, path_n, path_c, iteration=1000, learning_rate =1e-2, verbose=False):
    """ Entraîne le modèle 
    args:
        X_train (np.array): données d'entrainement
        y_train (np.array): labels d'entrainement
        X_test (np.array): données de test
        y_test (np.array): labels de test
        model (Resaux): modèle à entraîner
        path_n (str): chemin pour sauvegarder le modèle
        path_c (str): chemin pour sauvegarder les courbes
        iteration (int): nombre d'itérations
        learning_rate (float): taux d'apprentissage
        returns:
            model (Resaux): modèle entraîné"""

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    model.train(X_train, y_train, X_test, y_test, path_c, learning_rate, iteration)
    model.save(path_n, path_c, verbose)
    return model

def model_charge(path_n, verbose=False):
    """ Charge un modèle depuis un fichier
    args:
        path_n (str): chemin du fichier du modèle
    returns:
        model (Resaux): modèle chargé"""
    
    try:
        # Vérifier si le fichier existe et a une taille correcte
        if not os.path.exists(path_n):
            raise FileNotFoundError(f"Fichier {path_n} introuvable")
        
        file_size = os.path.getsize(path_n)
        if file_size == 0:
            raise ValueError(f"Fichier {path_n} est vide")
            
        print(f"DEBUG: Chargement depuis {path_n} (taille: {file_size} octets)")
        
        model = Resaux()
        model.load(path_n, verbose)
        return model
        
    except Exception as e:
        print(f"ERREUR model_charge: {repr(e)}")
        raise

# Outils d'analyse

def analyse_pre_process(df):
    ''' Analyse le dataset avant le prétraitement '''
    print()
    # Affiche les premières lignes du DataFrame
    print(df.head())
    print()
    # Affiche des informations sur le DataFrame
    print(df.info())
    print()
    # Statistiques descriptives
    print(df.describe())
    print()
    # Vérifie les valeurs manquantes
    print(df.isna().sum()/df.shape[0])
    print()

def analyse_post_process(X_train, y_train, X_test, y_test):
    '''
    Analyse le dataset après le prétraitement 
    Affiche les dimensions, types de données, et quelques exemples de données.
    args:
        X_train (np.array): données d'entrainement 
        y_train (np.array): labels d'entrainement
        X_test (np.array): données de test
        y_test (np.array): labels de test
    '''
    
    # Affiche les dimensions des ensembles de données
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Affiche les types de données et quelques exemples
    print("\nX_train data types:\n", pd.DataFrame(X_train).dtypes)
    print("\nFirst 5 rows of X_train:\n", pd.DataFrame(X_train).head())
    print("\ny_train data types:\n", pd.DataFrame(y_train).dtypes)
    print("\nFirst 5 rows of y_train:\n", pd.DataFrame(y_train).head())

def matrice_confusion(y_true, y_pred, classes, path, title='Matrice de confusion', cmap=plt.cm.Blues, verbose=False):
    """ Crée et sauvegarde la matrice de confusion 
    args:
        y_true (np.array): labels réels
        y_pred (np.array): labels prédits
        classes (list): liste des classes
        path (str): chemin pour sauvegarder la matrice
        title (str): titre du graphique
        cmap: colormap pour la matrice
    """
    # Calcul et sauvegarde de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=cmap)
    plt.title(title)
    plt.savefig(path)
    plt.close()
    if verbose:
        print("Matrice de confusion sauvegardée dans", path)

def graphique_residus(y_true, y_pred, path, title='Graphique des Résidus'):
    """
    Affiche un graphique des résidus pour un modèle de régression.
    args:
        y_true (np.array): valeurs réelles
        y_pred (np.array): valeurs prédites
        path (str): chemin pour sauvegarder le graphique
        title (str): titre du graphique

    """
    # Calcul des résidus
    residuals = y_true - y_pred
    
    # Tracé du graphique des résidus
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Valeurs Prédites')
    plt.ylabel('Résidus (Erreurs)')
    plt.title(title)
    plt.grid(True)
    # Sauvegarde du graphique
    plt.savefig(path)
    plt.close()
    print("Graphique des résidus sauvegardé dans", path)

def affichage_perf(X_train, y_train, X_test, y_test, model, path, qt=None):
    """ Affiche automatiquement les métriques selon le type du modèle (régression, binaire, ou multi-classes) 
    args:
        X_train (np.array): données d'entrainement
        y_train (np.array): labels d'entrainement
        X_test (np.array): données de test
        y_test (np.array): labels de test
        model (Resaux): modèle à évaluer
        path (str): chemin pour sauvegarder les graphiques
        qt (QuantileTransformer): transformateur pour la régression
    """

    # Aplatir les labels
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

        graphique_residus(y_test_true, pred_test, path.replace('curve', 'residuals_graph'), title='Graphique des Résidus - Test')

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

        matrice_confusion(y_test_bin, pred_test, [0, 1], f"{path.replace('curve', 'confusion_matrix')}", title='Matrice de confusion - Test')

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

        class_labels = np.unique(np.concatenate((y_test_int, pred_test)))
        class_names = [str(c) for c in class_labels]
        matrice_confusion(y_test_int, pred_test, class_names, f"{path.replace('curve', 'confusion_matrix')}", title='Matrice de confusion - Test')

def val_evolution(model, input_row, modifiable_features, modifiable_indices, features, nb_iter=30):
    """ Utilise l'optimisation bayésienne pour trouver les meilleures modifications des caractéristiques modifiables 
        afin de maximiser la prédiction du modèle. (utilisé pour l'IA sur la qualité du sommeil)
    args:
        model (Resaux): modèle entraîné
        input_row (np.array): ligne d'entrée à modifier
        modifiable_features (list): liste des caractéristiques modifiables
        modifiable_indices (list): indices des caractéristiques modifiables dans l'entrée
        features (list): liste de toutes les caractéristiques
        nb_iter (int): nombre d'itérations pour l'optimisation bayésienne
    """
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
        'Physical Activity Level': (0 / 360, 240 / 360),  # 0 à 240 min activité
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
