import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox, QHBoxLayout, QTextEdit, QCheckBox, QLineEdit, QFormLayout
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, pyqtSignal, QObject

# Importation de mes modules
import Core.training.AI_training_Quality as ATQ
import Core.training.AI_training_Trouble as ATT
import UI.app as APP

class EmittingStream(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(text)
        QApplication.processEvents()

    def flush(self):
        pass # Nécessaire pour certains environnements

class MainMenu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Menu Principal")
        self.setGeometry(100, 100, 1000, 800)

        # Main Layout (
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(25)

        # --- Left Panel ---
        left_layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Choisissez un script à lancer :")
        title_label.setObjectName("titleLabel")
        left_layout.addWidget(title_label)
        
        # Buttons
        self.btn_app = QPushButton("Lancer l'Application UI")
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
        
        # Checkboxes
        self.cb_c = QCheckBox()
        self.cb_t = QCheckBox()
        self.cb_keep_paths = QCheckBox()
        self.cb_verbose = QCheckBox()

        # Text fields
        self.default_paths = {
            'quality': {
                'path_n': 'TIPE/Code/Saves/save_sleep_quality.pkl',
                'path_c': 'TIPE/Code/Saves_Curves/curve_sleep_quality.png'
                  },
            'trouble': {
                'path_n': 'TIPE/Code/Saves/save_sleep_trouble.pkl',
                'path_c': 'TIPE/Code/Saves_Curves/curve_sleep_trouble.png'
                 }
        }
        
        self.line_path_n = QLineEdit()
        self.line_path_c = QLineEdit()

        param_form.addRow("Création nouvelle IA:", self.cb_c)
        param_form.addRow("Mode Entraînement:", self.cb_t)
        param_form.addRow("Conserver les chemins:", self.cb_keep_paths)
        param_form.addRow("Verbose:", self.cb_verbose)
        param_form.addRow("Chemin de sauvegarde:", self.line_path_n)
        param_form.addRow("Chemin des courbes:", self.line_path_c)

        left_layout.addLayout(param_form)
        left_layout.addStretch(1)
        main_layout.addLayout(left_layout, 1)

        # --- Right Panel: Image and Console ---
        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)

        # Image Viewer
        self.image_dir = 'TIPE/Code/Saves_Curves'
        self.image_combo = QComboBox()
        if os.path.exists(self.image_dir):
            self.image_combo.addItems([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.image_label = QLabel("Sélectionnez une image pour l'afficher")
        self.image_label.setObjectName("imageLabel")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(600, 400)
        self.image_combo.currentIndexChanged.connect(self.display_image)

        right_layout.addWidget(self.image_combo)
        right_layout.addWidget(self.image_label)
        self.display_image()

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

    def display_image(self):
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
        self.console_output.insertPlainText(text)

    def update_form_paths(self, script_type):
        """Met à jour les chemins dans le formulaire en fonction du type de script."""
        paths = self.default_paths.get(script_type, {})
        if not self.line_path_n.text():
            self.line_path_n.setText(paths.get('path_n', ''))
        if not self.line_path_c.text():
            self.line_path_c.setText(paths.get('path_c', ''))

    def clear_form_paths(self):
        """Efface les champs de chemin du formulaire."""
        self.line_path_n.clear()
        self.line_path_c.clear()

    def run_app_script(self):
        # Lancer l'application UI et cacher le menu
        app_window = APP.MainWindow()
        app_window.show()
        self.hide()

    def run_atq_script(self):
        # Lancer le script de training qualité
        self.console_output.clear()
        print("\nLancement du script ATQ...\n")

        # Récupérer les valeurs du formulaire
        self.update_form_paths('quality')
        bool_c = self.cb_c.isChecked()
        bool_t = self.cb_t.isChecked()
        path_n = self.line_path_n.text()
        path_c = self.line_path_c.text()
        verbose = self.cb_verbose.isChecked()
        
        # Lancer le script avec les paramètres récupérés
        ATQ.main_quality_of_sleep(bool_c, bool_t, path_n, path_c, verbose)

        if not self.cb_keep_paths.isChecked():
            self.clear_form_paths()

        self.display_image() 

    def run_att_script(self):
        # Lancer le script de training trouble
        self.console_output.clear()

        # Récupérer les valeurs du formulaire
        self.update_form_paths('trouble')
        bool_c = self.cb_c.isChecked()
        bool_t = self.cb_t.isChecked()
        path_n = self.line_path_n.text()
        path_c = self.line_path_c.text()
        verbose = self.cb_verbose.isChecked()

        print("\nLancement du script ATT...\n")
        ATT.main_sleep_trouble(bool_c, bool_t, path_n, path_c, verbose)

        if not self.cb_keep_paths.isChecked():
            self.clear_form_paths()

        self.display_image()

def run_menu():
    app = QApplication(sys.argv)
    menu = MainMenu()
    menu.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_menu()