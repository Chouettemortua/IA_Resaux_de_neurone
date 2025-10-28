
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def init_explainer(model, X_ref, verbose=False):
    """
    Initialise un explainer SHAP pour un modèle donné.

    Args:
        model: un Modèle entraîné (supporté par SHAP)
        X_ref (pd.DataFrame ou np.ndarray): Jeu de données de référence pour le SHAP
        verbose (bool): Si True, affiche des messages de progression

    Returns:
        explainer: objet shap.Explainer prêt à l'emploi
    """
    if verbose: print("[SHAP] Initialisation de l'explainer...")
    try:
        explainer = shap.Explainer(lambda X : model.predict(X), X_ref)
        if verbose: print("[SHAP] Explainer initialisé avec succès.")
        return explainer
    except Exception as e:
        print(f"[SHAP] Erreur lors de l'initialisation de l'explainer : {e}")
        return None


def compute_shap_values(explainer, X_sample, sample_size=None, verbose=False):
    """
    Calcule les valeurs SHAP pour un échantillon.

    Args:
        explainer: Explainer shap créé par init_explainer()
        X_sample (pd.DataFrame ou np.ndarray): Données à analyser
        sample_size (int): Nombre d'exemples à utiliser (None pour tout)
        verbose (bool): Si True, affiche des messages de progression

    Returns:
        shap_values: tableau de valeurs SHAP
    """
    if sample_size not in (None, -1) and len(X_sample) > sample_size:
        X_sample = X_sample.sample(sample_size, random_state=42)
        if verbose: print(f"[SHAP] Sous-échantillon de {sample_size} lignes sélectionné.")
    
    if verbose: print(f"[SHAP] Calcul des valeurs SHAP pour {len(X_sample)} exemples...")
    shap_values = explainer(X_sample)
    if verbose: print("[SHAP] Calcul terminé.")
    return shap_values


def plot_summary(shap_values, X_sample, save_path=None, show=False, max_display=10, verbose=False):
    """
    Trace et sauvegarde le summary plot SHAP.

    Args:
        shap_values: Valeurs SHAP calculées (sortie de shap_values)
        X_sample: Données associées
        save_path (str): Chemin du fichier PNG à sauvegarder
        show (bool): Si True, affiche le graphique dans une fenêtre (doit avoir un environnement graphique valide)
        max_display (int): Nombre max de features affichées
        verbose (bool): Si True, affiche des messages de progression
    """
    if verbose: print("[SHAP] Génération du summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=max_display)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        if verbose: print(f"[SHAP] Summary plot sauvegardé à : {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_dependence(feature_name, shap_values, X_sample, save_path=None, show=False, verbose=False):
    """
    Trace et sauvegarde un graphe de dépendance SHAP pour une feature donnée.

    Args:
        feature_name (str): Nom de la variable à visualiser
        shap_values: Valeurs SHAP calculées
        X_sample: Données associées
        save_path (str): Chemin du fichier PNG à sauvegarder
        show (bool): Si True, affiche le graphique
        verbose (bool): Si True, affiche des messages de progression
    """
    if feature_name not in X_sample.columns:
        print(f"[SHAP] La feature '{feature_name}' n'existe pas dans les données.")
        return
    
    if verbose: print(f"[SHAP] Génération du dependence plot pour '{feature_name}'...")
    plt.figure()
    shap.dependence_plot(feature_name, shap_values.values, X_sample, show=False)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        if verbose: print(f"[SHAP] Dependence plot sauvegardé à : {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_force(explainer, shap_values, X, index=0, save_path=None, show=False, verbose=False):
    """Trace un force plot SHAP pour une ligne donnée.
    Args:
        explainer: Explainer SHAP utilisé pour générer les valeurs
        shap_values: Valeurs SHAP calculées
        X (pd.DataFrame): Données associées
        index (int): Index de la ligne à visualiser
        save_path (str): Chemin du fichier PNG à sauvegarder
        show (bool): Si True, affiche le graphique
    """

    try:
        # Récupération de expected_value
        expected_value = getattr(explainer, "expected_value", None)
        if expected_value is None:
            expected_value = getattr(explainer, "expected_values", None)

        if expected_value is None:
            if verbose: print("[SHAP] Avertissement : impossible de récupérer expected_value.")
            expected_value = 0  # fallback neutre si expected_value absent

        # Création du graphique
        shap_fig = shap.force_plot(
            expected_value,
            shap_values.values[index, :],
            X.iloc[index, :],
            matplotlib=True
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        if show:
            plt.show()

    except Exception as e:
        print(f"[SHAP] Erreur lors du tracé du force plot : {e}")
        

def shap_analysis(model, X, save_dir, sample_size=None, verbose=False, immutable_features=None):
    """
    Lance une analyse SHAP complète (summary + dépendance + force plot) pour un modèle donné.

    Args:
        model: Modèle IA (déjà entraîné)
        X (pd.DataFrame): Données d'entrée pour le calcul SHAP
        save_dir (str): Répertoire de sauvegarde des graphes
        sample_size (int | None): Taille max de l'échantillon pour accélérer le calcul (None = tout)
        verbose (bool): Afficher les logs dans la console
        immutable_features (list[str] | None): Liste des variables inchangeables à exclure des graphes
    """
    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print("\n[SHAP] === Début de l'analyse SHAP ===")

    # Sous-échantillonnage éventuel
    if sample_size not in (None, -1) and len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
        if verbose:
            print(f"[SHAP] Sous-échantillon de {sample_size} lignes sélectionné.")
    else:
        X_sample = X

    # Initialisation de l'explainer
    explainer = init_explainer(model, X_sample)
    if explainer is None:
        print("[SHAP] Impossible d'initialiser l'explainer, arrêt.")
        return None

    # Calcul des valeurs SHAP
    shap_values = compute_shap_values(explainer, X_sample)

    # Exclure les features inchangeables si demandées
    if immutable_features is not None:
        valid_features = [c for c in X_sample.columns if c not in immutable_features]
        X_display = X_sample[valid_features]
        shap_display = shap_values[:, [i for i, c in enumerate(X_sample.columns) if c in valid_features]]
        if verbose:
            print(f"[SHAP] {len(immutable_features)} features inchangeables exclues de l'affichage.")
    else:
        X_display = X_sample
        shap_display = shap_values

    # Tracé et sauvegarde des graphes
    summary_path = os.path.join(save_dir, "shap_summary.png")
    plot_summary(shap_display, X_display, save_path=summary_path, show=False)

    # Deux plots de dépendance sur les features les plus importantes (hors inchangeables)
    try:
        feature_importance = np.abs(shap_display.values).mean(axis=0)
        top_features = pd.Series(feature_importance, index=X_display.columns).sort_values(ascending=False).head(2)
        for feat in top_features.index:
            dep_path = os.path.join(save_dir, f"shap_dependence_{feat}.png")
            plot_dependence(feat, shap_display, X_display, save_path=dep_path, show=False)
    except Exception as e:
        print(f"[SHAP] Erreur lors du tracé des dépendances : {e}")

    # Force plot pour la première ligne (visualisation locale complète)
    force_path = os.path.join(save_dir, "shap_force.png")
    plot_force(explainer, shap_values, X_sample, index=0, save_path=force_path, show=False)

    if verbose:
        print("[SHAP] === Analyse SHAP terminée avec succès ===\n")

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "summary_path": summary_path,
        "force_path": force_path,
        "used_features": list(X_display.columns)
    }


def suggest_improvements(explainer, shap_values, X, index=-1, immutable_features=None, top_n=3, feature_directions=None, verbose=True):
    """
    Génère des conseils personnalisés à partir des valeurs SHAP d'un individu.

    Args:
        explainer: Explainer SHAP utilisé pour générer les valeurs
        shap_values: Valeurs SHAP pour tout le dataset
        X (pd.DataFrame): Données d'entrée associées
        index (int): Ligne à analyser (par défaut -1 = dernière)
        immutable_features (list[str]): Variables à ignorer dans les suggestions
        top_n (int): Nombre max de recommandations à renvoyer
        feature_directions (dict[str, str]): Indique le sens "positif" d'amélioration attendu
            ex: {"feature1": "higher", "feature2": "lower"}
        verbose (bool): Affiche les logs si True

    Returns:
        list[dict]: Liste de suggestions du type :
            [
                {"feature": "Stress", "impact": -0.42, "advice": "Réduire le stress"},
                ...
            ]
    """

    # Récupération des features et valeurs SHAP pour la ligne ciblée
    row_shap = shap_values.values[index]
    row_data = X.iloc[index]

    # Exclusion des features inchangeables
    features = X.columns
    if immutable_features:
        valid_idx = [i for i, f in enumerate(features) if f not in immutable_features]
        row_shap = row_shap[valid_idx]
        row_data = row_data[valid_idx]
        features = [f for f in features if f not in immutable_features]
        if verbose:
            print(f"[SHAP] {len(immutable_features)} variables inchangeables exclues des suggestions.")

    # Tri des features par impact (valeurs SHAP les plus négatives)
    shap_importance = np.argsort(row_shap)
    negative_idx = [i for i in shap_importance if row_shap[i] < 0]

    suggestions = []
    for i in negative_idx[:top_n]:
        feat = features[i]
        val = row_data[i]
        impact = row_shap[i]
        direction = feature_directions.get(feat) if feature_directions else None

        # Détermination du sens du conseil
        if direction == "lower":
            advice = f"Réduire {feat.lower()} (actuel: {val:.2f})"
        elif direction == "higher":
            advice = f"Augmenter {feat.lower()} (actuel: {val:.2f})"
        else:
            # Pas d'indication sur le sens → texte neutre
            advice = f"Ajuster {feat.lower()} (impact: {impact:.3f})"

        suggestions.append({
            "feature": feat,
            "value": float(val),
            "impact": float(impact),
            "advice": advice
        })

    if verbose:
        print(f"\n[SHAP] Suggestions pour l'utilisateur #{index}:")
        for s in suggestions:
            print(f"  • {s['advice']} | Impact {s['impact']:+.3f}")

    return suggestions