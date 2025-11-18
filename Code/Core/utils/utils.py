import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys


def courbe_perf(sleep, path, bool_p=True):
    """ Sauvegarde les courbes de perte et de performance dans un fichier donné.
    Args:
        sleep: Modèle entraîné dont on veut tracer les courbes.
        path: Chemin du fichier où sauvegarder les courbes.
        bool_p: Booléen pour afficher un message de confirmation.
    """
    plt.figure(figsize=(12, 4))

    # Titre et label dynamique selon le type de modèle
    if sleep.is_regression:
        acc_label = "R²"
        acc_title = "Courbe de R²"
        acc_ylabel = "R² score"
    else:
        acc_label = "Précision"
        acc_title = "Courbe de Précision"
        acc_ylabel = "Précision (%)"

    # Axe des x dépendant de partialsteps
    if sleep.partialsteps is not None:
        x_values = np.arange(0, len(sleep.L) * sleep.partialsteps, sleep.partialsteps)   
    else:
        x_values = range(len(sleep.L))

    # Perte (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(x_values, sleep.L, label="Perte d'entraînement")
    plt.plot(x_values, sleep.L_t, label="Perte de test")
    plt.legend()
    plt.title("Courbe de perte")
    plt.xlabel("Itérations")
    plt.ylabel("Perte (écart label-prédiction)")

    # Performance (Accuracy ou R²)
    plt.subplot(1, 2, 2)
    plt.plot(x_values, sleep.acc, label=f"Entraînement {acc_label}")
    plt.plot(x_values, sleep.acc_t, label=f"Test {acc_label}")
    plt.legend()
    plt.title(acc_title)
    plt.xlabel("Itérations")
    plt.ylabel(acc_ylabel)

    plt.tight_layout()
    # Sauvegarde et fermeture de la figure
    plt.savefig(path)
    plt.close()

    if bool_p:
        print("Courbes sauvegardées dans", path)

def get_base_path(file_path=None):
    # Vérifie si l'application est en mode "bundle" (compilée avec PyInstaller)
    if getattr(sys, 'frozen', False):
        # Si 'frozen' est True, on est dans l'exécutable
        # sys._MEIPASS pointe vers le répertoire temporaire où PyInstaller a décompressé les fichiers.
        # C'est l'emplacement pour lire les fichiers INCLUS dans le bundle.
        return os.path.dirname(os.path.abspath(sys.executable))
    else:
        # Si 'frozen' est False, on est en mode interprété normal.
        # Retourne le répertoire du script Python en cours d'exécution.
        if file_path is None:
            file_path = os.path.abspath(__file__)
        return os.path.dirname(file_path)
    
