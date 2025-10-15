import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, 
    QLabel, QComboBox, QHBoxLayout, QTextEdit, QCheckBox, QLineEdit, 
    QFormLayout, QSpinBox, QProgressBar)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread

# Importation de mes modules
import Core.training.AI_training as AT
import app_desk as APP
from Core.utils.utils import get_base_path

class EmittingStream(QObject):
    """ Classe pour rediriger la sortie standard vers une QTextEdit. """
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(text)
        QApplication.processEvents()

    def flush(self):
        pass 

class MainMenu(QMainWindow):
    def __init__(self):
        """ Initialise le menu principal de l'application. """

        super().__init__()
        # Configuration de la fenêtre principale
        self.setWindowTitle("Menu Principal")
        self.setGeometry(100, 100, 1000, 800)

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(25)

        # --- Partie Gauche: Bouton et formulaire ---
        left_layout = QVBoxLayout()
        
        # Titre
        title_label = QLabel("Choisissez un script à lancer :")
        title_label.setObjectName("titleLabel")
        left_layout.addWidget(title_label)
        
        # Buttons
        self.btn_app = QPushButton("Lancer l'Application Desktop")
        self.btn_atq = QPushButton("Lancer le script de Training Qualité")
        self.btn_att = QPushButton("Lancer le script de Training Trouble")

        self.btn_app.clicked.connect(self.run_app_script)
        self.btn_atq.clicked.connect(self.run_atq_script)
        self.btn_att.clicked.connect(self.run_att_script)

        left_layout.addWidget(self.btn_app)
        left_layout.addWidget(self.btn_atq)
        left_layout.addWidget(self.btn_att)
        left_layout.addSpacing(25)

        # Form Layout
        param_form = QFormLayout()
        param_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        param_form.setFormAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        param_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        # Line Edits pour les chemins
        self.line_path_n = QLineEdit()
        self.line_path_c = QLineEdit()

        # Checkboxes
        self.cb_c = QCheckBox()
        self.cb_t = QCheckBox()
        self.cb_keep_paths = QCheckBox()
        self.cb_verbose = QCheckBox()

        # Spin Box pour le nombre d'itérations
        self.line_nb_iter = QSpinBox()
        self.line_nb_iter.setRange(0, 100000)
        self.line_nb_iter.setSingleStep(1000)
        self.line_nb_iter.setValue(1000)

        # Ajout les éléments aux formulaires

        param_form.addRow("Création nouvelle IA:", self.cb_c)
        param_form.addRow("Mode Entraînement:", self.cb_t)
        param_form.addRow("Conserver les chemins:", self.cb_keep_paths)
        param_form.addRow("Verbose:", self.cb_verbose)
        param_form.addRow("Nombres d'iteration d'entrainement", self.line_nb_iter)
        param_form.addRow("Chemin de sauvegarde:", self.line_path_n)
        param_form.addRow("Chemin des courbes:", self.line_path_c)

        left_layout.addLayout(param_form)
        left_layout.addStretch(1)
        main_layout.addLayout(left_layout, 1)

        # --- Partie Droite: Image et Console ---
        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)

        # Définition des chemins par défaut des modèles et des images
        file_dir = os.path.abspath(__file__)
        parent_dir = get_base_path(file_dir)
        model_quality_path = os.path.join(parent_dir, 'Saves', 'save_sleep_quality.pkl')
        model_trouble_path = os.path.join(parent_dir, 'Saves', 'save_sleep_trouble.pkl')
        parent_dir_im = os.path.join(parent_dir, 'Saves_Curves')
        model_quality_curve_path = os.path.join(parent_dir_im, 'curve_sleep_quality.png')
        model_trouble_curve_path = os.path.join(parent_dir_im, 'curve_sleep_trouble.png')

        # Visualisation des images
        self.image_dir = parent_dir_im # Répertoire des images
        self.image_combo = QComboBox()  # ComboBox pour sélectionner les images
        if os.path.exists(self.image_dir): # Vérifie si le répertoire existe
            self.image_combo.addItems([f for f in os.listdir(self.image_dir) if f.endswith('.png')]) # Ajoute les fichiers PNG
        self.image_label = QLabel("Sélectionnez une image pour l'afficher") 
        self.image_label.setObjectName("imageLabel") 
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        self.image_label.setFixedSize(600, 400)
        self.image_combo.currentIndexChanged.connect(self.display_image) 

        right_layout.addWidget(self.image_combo)
        right_layout.addWidget(self.image_label)
        self.display_image()

        # Progression Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #dcdfe6;
                border-radius: 5px;
                text-align: center;
                background-color: #f5f5f5;
            }
            QProgressBar::chunk {
                background-color: #5d9cec;
                border-radius: 4px;
            }
        """)
        right_layout.addWidget(self.progress_bar)

        # Console
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setFixedHeight(250)
        self.console_output.setObjectName("consoleOutput")
        
        right_layout.addWidget(self.console_output)
        main_layout.addLayout(right_layout, 2)

        # Redirection de la sortie standard vers la console intégrée
        self.stdout_redirect = EmittingStream()
        self.stdout_redirect.text_written.connect(self.update_console)
        sys.stdout = self.stdout_redirect

        # print of paths for debug
        #print(f"Debug: parent_dir = {parent_dir}")
        #print(f"Debug: parent_dir_im = {parent_dir_im}")
        #print(f"Debug: model_quality_path = {model_quality_path}")
        #print(f"Debug: model_trouble_path = {model_trouble_path}")
        #print(f"Debug: model_quality_curve_path = {model_quality_curve_path}")
        #print(f"Debug: model_trouble_curve_path = {model_trouble_curve_path}")

        # Definition des chemins par défaut
        self.default_paths = {
            'quality': {
                'path_n': model_quality_path,
                'path_c': model_quality_curve_path
                  },
            'trouble': {
                'path_n': model_trouble_path,
                'path_c': model_trouble_curve_path
                 }
        }

        # --- Stylesheet ---
        self.setStyleSheet("""
            * {
                font-family: Arial, sans-serif;
            }
            QMainWindow {
                background-color: #f0f4f8;
            }
            #titleLabel {
                font-size: 20px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 15px;
            }
            QPushButton {
                background-color: #5d9cec;
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                border-radius: 6px;
                margin-bottom: 10px;
            }
            QPushButton:hover {
                background-color: #4a8cdb;
            }
            QComboBox, QLineEdit, QTextEdit {
                background-color: #ffffff;
                border: 1px solid #dcdfe6;
                padding: 8px;
                border-radius: 4px;
            }
            #imageLabel {
                border: 1px solid #dcdfe6;
                background-color: #e9eff4;
            }
            #consoleOutput {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: none;
                border-radius: 4px;
                font-family: Consolas, monospace;
            }
            QLabel {
                font-size: 14px;
                color: #34495e;
            }
        """)

    def update_progress_bar(self, value):
        """ Met à jour la barre de progression. """
        self.progress_bar.setValue(value)

    def display_image(self):
        """ Affiche l'image sélectionnée dans le QLabel. """
        selected_file = self.image_combo.currentText()
        if selected_file:

            image_path = os.path.join(self.image_dir, selected_file)
            
            pixmap = QPixmap(image_path)
            
            if pixmap.isNull():
                print(f"Erreur : le fichier {selected_file} n'a pas pu être chargé. Le pixmap est null.")
            else:
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(),
                                                        Qt.AspectRatioMode.KeepAspectRatio,
                                                        Qt.TransformationMode.SmoothTransformation)) 

    def update_console(self, text):
        """ Met à jour la console intégrée avec le texte fourni. """
        self.console_output.insertPlainText(text)

    def update_form_paths(self, script_type):
        """Met à jour les chemins dans le formulaire en fonction du type de script."""
        paths = self.default_paths.get(script_type, {})
        if not self.line_path_n.text():
            self.line_path_n.setText(paths.get('path_n', ''))
        if not self.line_path_c.text():
            self.line_path_c.setText(paths.get('path_c', ''))

    def set_buttons_enabled(self, enabled):
        """Active ou désactive les boutons du menu principal.
         Args:
             enabled (bool): True pour activer les boutons, False pour les désactiver.
         """
        self.btn_app.setEnabled(enabled)
        self.btn_atq.setEnabled(enabled) 
        self.btn_att.setEnabled(enabled)

    def clear_form_paths(self):
        """Efface les champs de chemin du formulaire."""
        self.line_path_n.clear()
        self.line_path_c.clear()

    def run_app_script(self):
        """ Lancer l'application UI """
        print("Lancement de l'application Desktop...")
        app_window = APP.MainWindow()
        app_window.show()
        self.hide()

    def run_atq_script(self):
        """ Lancer le script de training qualité """
        self.set_buttons_enabled(False)  # Désactive les boutons pendant l'exécution
        self.console_output.clear()
        print("\nLancement du script ATQ...\n")

        # Récupérer les valeurs du formulaire 
        self.update_form_paths('quality')
        bool_c = self.cb_c.isChecked()
        bool_t = self.cb_t.isChecked()
        path_n = self.line_path_n.text()
        path_c = self.line_path_c.text()
        nb_iter = self.line_nb_iter.value()
        verbose = self.cb_verbose.isChecked()

        # Afficher l'image de courbe correspondante a l'IA en cours d'entrainement
        self.image_combo.setCurrentText(path_c.split('/')[-1])
        self.display_image()
        
        # Lancer le script dans un thread séparé
        self.thread = QThread()
        self.worker = AT.TrainingWorker("Q", bool_c, bool_t, path_n, path_c, nb_iter, verbose)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress_updated.connect(self.update_progress_bar)
        self.worker.curve_save.connect(self.display_image) # Mettre a jour l'image a chaque fois qu'elle est sauvegarder pendant l'entrainement 

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(lambda: self.set_buttons_enabled(True))  # Réactive les boutons après l'exécution

        self.thread.start()

        # Effacer les chemins si la case n'est pas cochée
        if not self.cb_keep_paths.isChecked():
            self.clear_form_paths()
        
        # Réinitialiser le nombre d'itérations
        self.line_nb_iter.setValue(1000) 

    def run_att_script(self):
        """ Lancer le script de training trouble """
        self.set_buttons_enabled(False)  # Désactive les boutons pendant l'exécution
        self.console_output.clear()

        # Récupérer les valeurs du formulaire
        self.update_form_paths('trouble')
        bool_c = self.cb_c.isChecked()
        bool_t = self.cb_t.isChecked()
        path_n = self.line_path_n.text()
        path_c = self.line_path_c.text()
        nb_iter = self.line_nb_iter.value()
        verbose = self.cb_verbose.isChecked()

        # Afficher l'image de courbe correspondante a l'IA en cours d'entrainement
        self.image_combo.setCurrentText(path_c.split('/')[-1])
        self.display_image()

        # Lancer le script dans un thread séparé
        print("\nLancement du script ATT...\n")
        self.thread = QThread()
        self.worker = AT.TrainingWorker("T", bool_c, bool_t, path_n, path_c, nb_iter, verbose)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        # Connecter les signaux du worker aux slots appropriés
        self.worker.progress_updated.connect(self.update_progress_bar) # Mettre a jour la barre de progression
        self.worker.curve_save.connect(self.display_image) # Mettre a jour l'image a chaque fois qu'elle est sauvegarder pendant l'entrainement 
        
        self.worker.finished.connect(self.thread.quit) # Quitter le thread après la fin
        self.worker.finished.connect(self.worker.deleteLater) # Nettoyer le worker après la fin
        self.thread.finished.connect(self.thread.deleteLater) # Nettoyer le thread après la fin
        self.thread.finished.connect(lambda: self.set_buttons_enabled(True))  # Réactive les boutons après l'exécution

        self.thread.start() # Démarrer le thread

        # Mettre à jour l'image de courbe après l'entraînement
        self.display_image()

        # Effacer les chemins si la case n'est pas cochée
        if not self.cb_keep_paths.isChecked():
            self.clear_form_paths()

        # Réinitialiser le nombre d'itérations
        self.line_nb_iter.setValue(1000)

def run_menu():
    """ Lance le menu principal de l'application. """
    app = QApplication(sys.argv)
    menu = MainMenu()
    menu.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_menu()
    
