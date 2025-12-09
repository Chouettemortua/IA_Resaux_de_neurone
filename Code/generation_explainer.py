
from Core.utils.shap_module import init_explainer
from Core.training.training_utils import load, model_charge
from Core.preprocessing.preprocessing import preprocecing
import pickle
import os

def X_reference_generation(df, model_type):
    """ Génère un ensemble de référence pour SHAP en fonction du type de modèle.
    Args:
        df: DataFrame contenant les données.
        model_type: Type de modèle (Q pour qualité du someille et T pour les trouble).
    Returns:
        X_reference: Ensemble de référence pour SHAP.
    """
    if model_type == "T":
        _, _, X_test, _ = preprocecing(df, ['Sleep Disorder', 'Quality of Sleep'], y_normalisation=False)
    elif model_type == "Q":
        _, _, X_test, _ = preprocecing(df, ['Quality of Sleep', 'Sleep Disorder'], y_normalisation=False)
    else:
        raise ValueError("Type de modèle inconnu. Utilisez 'T' pour classification ou 'Q' pour régression.")
    
    return X_test

def path_by_type(saves_dir, model_type):
    """ Charge le modèle en fonction du type.
    Args:
        saves_dir: Répertoire de sauvegarde des modèles.
        model_type: Type de modèle (Q pour qualité du someille et T pour les trouble).
    Returns:
        model: Modèle chargé.
    """
    if model_type == "T":
        model_filename = os.path.join(saves_dir, 'save_sleep_trouble.pkl')
    elif model_type == "Q":
        model_filename = os.path.join(saves_dir, 'save_sleep_quality.pkl')
    else:
        raise ValueError("Type de modèle inconnu. Utilisez 'T' pour classification ou 'Q' pour régression.")
    return model_filename

def save_explainer(X_reference, model_path, filename):
    """ Sauvegarde l'explainer SHAP dans un fichier.
    Args:
        explainer: Explainer SHAP à sauvegarder.
        filename: Nom du fichier de sauvegarde.
    """
    with open(filename, 'wb') as f:
        pickle.dump({
            'X_reference': X_reference,
            'model_path': model_path
        }, f)
    

def load_explainer(filename):
    """ Charge un explainer SHAP depuis un fichier.
    Args:
        filename: Nom du fichier de sauvegarde.
    Returns:
        explainer: Explainer SHAP chargé.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        X_reference = data['X_reference']
        model_path = data['model_path']
    explainer = init_explainer(X_reference, model_charge(model_path))
    return explainer

if __name__ == "__main__":
    
    # Chargement du dataset
    global_dir = os.path.dirname(os.path.relpath(__file__))
    saves_dir = os.path.join(global_dir, 'Saves')
    data_dir = os.path.join(global_dir, 'Data')

    data = load(os.path.join(data_dir, 'Sleep_health_and_lifestyle_dataset.csv'))
    df = data.copy()

    for model_type in ['Q', 'T']:
        # Génération de l'ensemble de référence
        X_reference = X_reference_generation(df, model_type)

        # Chargement du modèle
        model_path = path_by_type(saves_dir, model_type)

        # Sauvegarde de l'explainer
        explainer_filename = os.path.join(saves_dir, f'shap_explainer_{model_type}.pkl')
        save_explainer(X_reference, model_path, explainer_filename)