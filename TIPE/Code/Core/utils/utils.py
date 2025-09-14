import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    # Sauvegarde et fermeture de la figure
    plt.savefig(path)
    plt.close()

    if bool_p:
        print("Courbes sauvegardées dans", path)

