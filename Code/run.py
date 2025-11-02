import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, 
    QLabel, QHBoxLayout, QTextEdit, QCheckBox, QLineEdit, 
    QFormLayout, QSpinBox, QProgressBar, QListWidget, QListWidgetItem)
from PyQt6.QtGui import QPixmap, QIcon, QFont, QFontMetrics
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QSize, QRect

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
        
        # Boutons
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
        self.cb_shap = QCheckBox()
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
        param_form.addRow("Analyse SHAP:", self.cb_shap)
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
        
        save_dir = os.path.join(parent_dir, 'Saves')
        model_quality_path = os.path.join(save_dir, 'save_sleep_quality.pkl')
        model_trouble_path = os.path.join(save_dir, 'save_sleep_trouble.pkl')
        parent_dir_im = os.path.join(parent_dir, 'Saves_Curves')
        parent_dir_im_q = os.path.join(parent_dir_im, 'Quality')
        parent_dir_im_t = os.path.join(parent_dir_im, 'Trouble')
        model_quality_curve_path = os.path.join(parent_dir_im_q, 'curve_sleep_quality.png')
        model_trouble_curve_path = os.path.join(parent_dir_im_t, 'curve_sleep_trouble.png')

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(parent_dir_im, exist_ok=True)
        os.makedirs(parent_dir_im_q, exist_ok=True)
        os.makedirs(parent_dir_im_t, exist_ok=True)

        self.image_dir = parent_dir_im
        self.image_list = QListWidget() 
        
        # Configuration de l'affichage en mode galerie horizontale
        self.image_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list.setFlow(QListWidget.Flow.LeftToRight) 
        self.image_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.image_list.setWordWrap(True) 
        self.image_list.setDragDropMode(QListWidget.DragDropMode.NoDragDrop) 
        self.image_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection) 
        
        # Variables de mesure et d'initialisation
        png_files = []
        max_text_width = 0
        icon_width = 120
        icon_height = 90
        
        # Récupérer la police de l'item pour une mesure précise
        item_font = QFont("Arial", 9)
        metrics = QFontMetrics(item_font)
        text_line_height = metrics.lineSpacing()
        
        if os.path.exists(self.image_dir):
            for root, _, files in os.walk(self.image_dir):
                for f in files:
                    if f.endswith('.png'):
                        # ... (collecte des chemins) ...
                        full_path = os.path.join(root, f)
                        relative_path = os.path.relpath(full_path, self.image_dir)
                        png_files.append(relative_path)
            
            png_files_sorted = sorted(png_files)

            # Calcul de la largeur maximale du texte
            for relative_path in png_files_sorted:
                text_width = metrics.horizontalAdvance(relative_path)
                if text_width > max_text_width:
                    max_text_width = text_width
            
            # Définir une largeur cellule fixe basée sur la largeur de l'icon
            fixed_cell_width = icon_width + 20 # Marge fixe pour le padding QSS
            
            max_required_height = 0
            
            # Utiliser QFontMetrics pour trouver la hauteur réelle (Word-wrap)
            metrics = QFontMetrics(item_font)
            
            for relative_path in png_files_sorted:
                # Simuler le Word Wrap à l'intérieur de la largeur FIXE de la cellule.
                # Nous laissons une petite marge de 10px (5px de chaque côté).
                text_rect = metrics.boundingRect(
                    QRect(0, 0, fixed_cell_width - 10, 1000), 
                    Qt.TextFlag.TextWordWrap, 
                    relative_path
                )
                
                # La hauteur totale est la hauteur de l'icône + la hauteur du texte wrappé + padding.
                current_item_height = icon_height + text_rect.height() + 4 # 4px pour padding QSS vertical

                if current_item_height > max_required_height:
                    max_required_height = current_item_height
            
            # Si la liste est vide, utiliser la valeur par défaut pour éviter les erreurs
            if max_required_height == 0:
                # Hauteur de base pour 2 lignes si la liste est vide
                max_required_height = icon_height + (2 * text_line_height) + 4 
            
            # Fixer la hauteur de la cellule à la valeur maximale trouvée
            final_cell_height = max_required_height
            
            # Application des tailles 
            self.image_list.setIconSize(QSize(icon_width, icon_height)) 
            
            # **CLÉ : Utiliser la hauteur maximale trouvée (final_cell_height)**
            self.image_list.setGridSize(QSize(fixed_cell_width, final_cell_height)) 
            
            self.image_list.setFixedHeight(final_cell_height) # Hauteur fixe pour éviter le redimensionnement vertical
            
            # Création et ajout des éléments avec miniatures
            for relative_path in png_files_sorted:
                full_path = os.path.join(self.image_dir, relative_path)
                
                # Charger la miniature
                pixmap = QPixmap(full_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        QSize(icon_width, icon_height), 
                        Qt.AspectRatioMode.KeepAspectRatio, 
                        Qt.TransformationMode.SmoothTransformation
                    )
                    icon = QIcon(scaled_pixmap)
                else:
                    icon = QIcon() 
                    
                # Créer l'élément de la liste
                item = QListWidgetItem(icon, relative_path)
                item.setFont(item_font)
                item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
                self.image_list.addItem(item)
            
            # Sélectionner le premier élément
            if self.image_list.count() > 0:
                self.image_list.setCurrentRow(0)


        self.image_label = QLabel("Sélectionnez une image pour l'afficher") 
        self.image_label.setObjectName("imageLabel") 
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        self.image_label.setFixedSize(600, 300)
        
        # Connexion au signal pour afficher l'image sélectionnée
        self.image_list.currentItemChanged.connect(self.display_image)

        right_layout.addWidget(self.image_list)
        right_layout.addWidget(self.image_label)
        
        # Premier appel à display_image (pour afficher la première image sélectionnée)
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

        # ... (Redirection de la sortie standard) ...
        sys.stdout = EmittingStream(text_written=self.update_console)
        sys.stderr = EmittingStream(text_written=self.update_console)

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
        # ... (Votre Stylesheet complet) ...
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
             QListWidget {
                 background-color: #ffffff;
                 border: 1px solid #dcdfe6;
                 border-radius: 4px;
                 outline: none;
                 padding: 2px; 
             }
             QListWidget::item {
                 background-color: #ebf5ff; 
                 padding: 1px 5px 1px 5px;
                 color: #34495e;
                 border: 0.25px solid #dcdfe6; 
                 border-radius: 4px;
             }
             QListWidget::item:selected {
                 background-color: #5d9cec; /* bleu vif */
                 padding: 1px 5px 1px 5px;
                 color: white; 
                 border: 0.25px solid #3c7fd6; /* bleue foncée */
                 border-radius: 4px;
             }
             QListWidget::item:hover {
                 background-color: #d8e6f8;
             }
         """)

    def update_progress_bar(self, value):
        """ Met à jour la barre de progression. """
        self.progress_bar.setValue(value)

    def display_image(self):
        """ Affiche l'image sélectionnée dans le QLabel. """

        # On récupère l'élément sélectionné dans la liste
        current_item = self.image_list.currentItem()

        if current_item is None:
            # Si aucun élément n'est sélectionné (cas de l'initialisation ou liste vide)
            self.image_label.setText("Sélectionnez une image pour l'afficher")
            return
        
        # Récupérer le nom du fichier sélectionné
        selected_file = current_item.text()

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
        sys.stdout = sys.__stdout__
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
        shap = self.cb_shap.isChecked()
        verbose = self.cb_verbose.isChecked()

        # Afficher l'image de courbe correspondante a l'IA en cours d'entrainement
        selected_path = os.path.relpath(path_c, self.image_dir)
        items = self.image_list.findItems(selected_path, Qt.MatchFlag.MatchExactly)
        if items:
            self.image_list.setCurrentItem(items[0])
            self.image_list.scrollToItem(items[0])
        self.display_image()
        
        # Lancer le script dans un thread séparé
        self.thread = QThread()
        self.worker = AT.TrainingWorker("Q", bool_c, bool_t, path_n, path_c, nb_iter, verbose, shap)

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
        shap = self.cb_shap.isChecked()
        verbose = self.cb_verbose.isChecked()

        # Afficher l'image de courbe correspondante a l'IA en cours d'entrainement
        selected_path = os.path.relpath(path_c, self.image_dir)
        items = self.image_list.findItems(selected_path, Qt.MatchFlag.MatchExactly)
        if items:
            self.image_list.setCurrentItem(items[0])
            self.image_list.scrollToItem(items[0])
        self.display_image()

        # Lancer le script dans un thread séparé
        print("\nLancement du script ATT...\n")
        self.thread = QThread()
        self.worker = AT.TrainingWorker("T", bool_c, bool_t, path_n, path_c, nb_iter, verbose, shap)

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

    def closeEvent(self, event):
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.stdout = sys.__stdout__
        super().closeEvent(event)

def run_menu():
    """ Lance le menu principal de l'application. """
    app = QApplication(sys.argv)
    menu = MainMenu()
    menu.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_menu()
    
