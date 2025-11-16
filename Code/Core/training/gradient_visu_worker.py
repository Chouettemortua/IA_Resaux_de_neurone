from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np
from .training_utils import model_charge

class GradientVisuWorker(QObject):
    result_ready = pyqtSignal(object, object, object)  # alphas, betas, Z_surface
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model_path, X, y, steps=25, scale=1.0):
        """ Initialise le Worker thread pour calculer les norme du gradient sur un cadrillage dans l'espace des paramètres.
        Args:
            model: Le modèle de réseau de neurones.
            X: Les données d'entrée.
            y: Les étiquettes cibles.
            steps: Nombre de points par axe dans le cadrillage.
            scale: Étendue des perturbations dans l'espace des paramètres.
        Returns:
            Émet un signal avec les matrices alphas, betas et Z_surface une fois le calcul terminé.
        """
        super().__init__()
        self.model_path = model_path
        self.X = X
        self.y = y
        self.steps = steps
        self.scale = scale

    def flatten_params(self, W_list, b_list):
        """ Aplati les listes de matrices de poids et de biais en un seul vecteur, tout en conservant les formes originales. 
         Args:
            W_list: Liste des matrices de poids.
            b_list: Liste des vecteurs de biais.
         Returns:
            flat: Vecteur aplati des paramètres.
            shapes: Liste des formes originales des matrices de poids et de biais."""
        flat = []
        shapes = []
        for W in W_list:
            shapes.append(("W", W.shape))
            flat.append(W.flatten())
        for b in b_list:
            shapes.append(("b", b.shape))
            flat.append(b.flatten())
        return np.concatenate(flat), shapes

    def unflatten_params(self, vector, shapes):
        """ Reconstruit les listes de matrices de poids et de biais à partir d'un vecteur aplati.
            Args:
                vector: Vecteur aplati des paramètres.
                shapes: Liste des formes originales des matrices de poids et de biais.
            Returns:
                W_list: Liste des matrices de poids reconstruites.
                b_list: Liste des vecteurs de biais reconstruits."""
        W_list = []
        b_list = []
        idx = 0
        for kind, shape in shapes:
            size = np.prod(shape)
            chunk = vector[idx:idx+size].reshape(shape)
            idx += size
            if kind == "W":
                W_list.append(chunk)
            else:
                b_list.append(chunk)
        return W_list, b_list

    def set_model_params(self, flat_vector, shapes):
        """ Met à jour les paramètres du modèle à partir d'un vecteur aplati.
            Args:
                flat_vector: Vecteur aplati des paramètres.
                shapes: Liste des formes originales des matrices de poids et de biais."""
        W_list, b_list = self.unflatten_params(flat_vector, shapes)
        self.model.W = W_list
        self.model.b = b_list

    def run(self):
        """ Exécute le calcul des normes du gradient sur le cadrillage dans l'espace des paramètres. """
        try:
            self.model = model_charge(self.model_path)
            base_W = [W.copy() for W in self.model.W]
            base_b = [b.copy() for b in self.model.b]
            base_vec, shapes = self.flatten_params(base_W, base_b)

            d1 = np.random.randn(*base_vec.shape)
            d2 = np.random.randn(*base_vec.shape)
            d1 /= np.linalg.norm(d1)
            d2 /= np.linalg.norm(d2)

            alphas = np.linspace(-self.scale, self.scale, self.steps)
            betas  = np.linspace(-self.scale, self.scale, self.steps)
            Z_surface = np.zeros((self.steps, self.steps))

            for i, a in enumerate(alphas):
                for j, b in enumerate(betas):
                    new_vec = base_vec + a*d1 + b*d2
                    self.set_model_params(new_vec, shapes)

                    A, Z = self.model.forward_propagation(self.X)
                    dW, dB = self.model.back_propagation(Z, A, self.X, self.y)

                    grad_norm = np.sqrt(sum(np.sum(dw**2) for dw in dW) + sum(np.sum(db**2) for db in dB))
                    Z_surface[i, j] = grad_norm

            self.set_model_params(base_vec, shapes)

            # Émet les signals une fois terminé
            self.result_ready.emit(alphas, betas, Z_surface)
            self.finished.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.finished.emit()
