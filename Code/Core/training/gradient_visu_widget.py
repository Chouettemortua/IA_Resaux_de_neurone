from PyQt6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from .gradient_visu_worker import GradientVisuWorker
import numpy as np

class GradientVisuWidget(QtWidgets.QDialog):
    def __init__(self, model_path, X, y, parent=None):
        """ Widget pour visualiser la norme du gradient dans l'espace des paramètres du modèle.
        Args:
            model: Le path de la sauvegarde du modèle de réseau de neurones.
            X: Les données d'entrée.
            y: Les étiquettes cibles.
        """
        super().__init__(parent)
        self.model_path = model_path
        self.X = X
        self.y = y

        layout = QtWidgets.QVBoxLayout(self)

        # Canvas matplotlib
        self.fig = Figure(figsize=(8,6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Contrôles sliders
        control_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(control_layout)

        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.scale_slider.setMinimum(1)
        self.scale_slider.setMaximum(1000)
        self.scale_slider.setValue(500)
        control_layout.addWidget(QtWidgets.QLabel("Scale"))
        control_layout.addWidget(self.scale_slider)

        self.steps_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.steps_slider.setMinimum(10)
        self.steps_slider.setMaximum(50)
        self.steps_slider.setValue(25)
        control_layout.addWidget(QtWidgets.QLabel("Steps"))
        control_layout.addWidget(self.steps_slider)

        # Bouton calcul
        self.calc_button = QtWidgets.QPushButton("Calculer Surface")
        layout.addWidget(self.calc_button)
        self.calc_button.clicked.connect(self.start_worker)

    def start_worker(self):
        """ Démarre le thread de calcul pour la norme du gradient. """
        steps = self.steps_slider.value()
        scale = self.scale_slider.value()/100.0

        # Désactiver bouton pour éviter double click
        self.calc_button.setEnabled(False)

        self.thread = QtCore.QThread()
        self.worker = GradientVisuWorker(self.model_path, self.X, self.y, steps=steps, scale=scale)
        self.worker.moveToThread(self.thread)

        self.worker.result_ready.connect(self.update_surface)
        self.worker.error_occurred.connect(self.handle_error)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def handle_error(self, error_message):
        """ Gère les erreurs survenues dans le thread de travail.
            Args:
                error_message: Message d'erreur à afficher.
        """
        QtWidgets.QMessageBox.critical(self, "Erreur", f"Une erreur est survenue lors du calcul :\n{error_message}")
        self.calc_button.setEnabled(True)

    def update_surface(self, alphas, betas, Z_surface):
        """ Met à jour la surface 3D avec les nouvelles données calculées.
            Args:
                alphas: Valeurs sur l'axe alpha.
                betas: Valeurs sur l'axe beta.
                Z_surface: Matrice des normes du gradient.
        """
        self.ax.clear()
        A_grid, B_grid = np.meshgrid(alphas, betas)
        self.ax.plot_surface(A_grid, B_grid, Z_surface, cmap='plasma')
        self.ax.set_xlabel("Direction 1 (α)")
        self.ax.set_ylabel("Direction 2 (β)")
        self.ax.set_zlabel("||Gradient||")
        self.ax.set_title("Gradient Norm Surface")
        self.canvas.draw()

        self.calc_button.setEnabled(True)
