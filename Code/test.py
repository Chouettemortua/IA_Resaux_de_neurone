'''import re
import csv

def parse_line(line):
    # Exemple de ligne :  
    # ΔSleep Duration = -1.20 | ΔPhysical Activity Level = -100.00 | ΔStress Level = 0.50 | ΔBMI Category = -0.20 | ΔBlood Pressure = 40.00 → prédiction : 0.556 (impact : -0.111)
    
    # Sépare la partie avant la flèche → et la partie après
    try:
        features_part, pred_part = line.split('→')
    except ValueError:
        return None  # ligne mal formée

    # Récupère la prédiction et l'impact via regex
    m = re.search(r"prédiction\s*:\s*([\d\.]+)\s*\(impact\s*:\s*([-+]?\d*\.?\d+)\)", pred_part)
    if not m:
        return None

    prediction = float(m.group(1))
    impact = float(m.group(2))

    # On ne garde que les lignes où impact ≠ 0
    if impact == 0:
        return None

    # Parse les features et leurs deltas
    features = features_part.split('|')
    parsed_features = []
    for feat in features:
        feat = feat.strip()
        # Format attendu : ΔFeature Name = value
        m_feat = re.match(r"Δ(.+?)\s*=\s*([-+]?\d*\.?\d+)", feat)
        if m_feat:
            name = m_feat.group(1).strip()
            val = float(m_feat.group(2))
            parsed_features.append((name, val))

    return parsed_features, prediction, impact

def process_file(input_path, output_path):
    all_rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            parsed = parse_line(line)
            if parsed:
                features, pred, impact = parsed
                # Pour CSV, on va aplatir features en colonnes : Feature_1, Delta_1, Feature_2, Delta_2, ...
                row = {}
                for i, (fname, fval) in enumerate(features, start=1):
                    row[f'Feature_{i}'] = fname
                    row[f'Delta_{i}'] = fval
                row['Prediction'] = pred
                row['Impact'] = impact
                all_rows.append(row)

    # Trie par impact décroissant
    all_rows.sort(key=lambda x: x['Impact'], reverse=True)

    # Trouver le max de features pour créer les colonnes CSV
    max_features = max(len([k for k in row if k.startswith('Feature_')]) for row in all_rows)

    # Créer les noms de colonnes dynamiquement
    columns = []
    for i in range(1, max_features + 1):
        columns.append(f'Feature_{i}')
        columns.append(f'Delta_{i}')
    columns.extend(['Prediction', 'Impact'])

    # Écriture CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in all_rows:
            # Remplir les colonnes absentes avec ''
            for col in columns:
                if col not in row:
                    row[col] = ''
            writer.writerow(row)

if __name__ == "__main__":
    input_txt_file = "TIPE/Saves/recommandations_sleep_quality.txt"
    output_csv_file = "TIPE/Saves/recommandations_sleep_quality.csv"
    process_file(input_txt_file, output_csv_file)
    print(f"Fichier CSV généré : {output_csv_file}")
#---------------------------------------------------------

import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QComboBox, QFormLayout, QLineEdit, QCheckBox, QTextEdit
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QTextCursor

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
        pass

class MainMenu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Menu Principal")
        self.setGeometry(100, 100, 1000, 800)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(base_dir, 'Core', 'training', 'graphs')

        self.default_paths = {
            'quality': {
                'path_n': 'data/quality/normal',
                'path_c': 'data/quality/calibration'
            },
            'trouble': {
                'path_n': 'data/trouble/normal',
                'path_c': 'data/trouble/calibration'
            }
        }

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        left_layout = QVBoxLayout()
        title_label = QLabel("Choisissez un script à lancer :")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        left_layout.addWidget(title_label)

        self.btn_app = QPushButton("Lancer l'Application UI")
        self.btn_atq = QPushButton("Lancer le script de Training Qualité")
        self.btn_att = QPushButton("Lancer le script de Training Trouble")

        self.btn_app.clicked.connect(self.run_app_script)
        self.btn_atq.clicked.connect(self.run_atq_script)
        self.btn_att.clicked.connect(self.run_att_script)

        left_layout.addWidget(self.btn_app)
        left_layout.addWidget(self.btn_atq)
        left_layout.addWidget(self.btn_att)

        param_form = QFormLayout()
        
        self.cb_c = QCheckBox("bool_c (true)")
        self.cb_c.setChecked(True)
        self.cb_t = QCheckBox("bool_t (true)")
        self.cb_t.setChecked(True)
        
        self.line_path_n = QLineEdit()
        self.line_path_c = QLineEdit()
        
        self.cb_verbose = QCheckBox("Verbose (detailed output)")
        
        # NOUVEAU : Case à cocher pour conserver les chemins
        self.cb_keep_paths = QCheckBox("Conserver les chemins")
        
        param_form.addRow("Mode C:", self.cb_c)
        param_form.addRow("Mode T:", self.cb_t)
        param_form.addRow("Chemin Normal:", self.line_path_n)
        param_form.addRow("Chemin Calibration:", self.line_path_c)
        param_form.addRow("Verbose:", self.cb_verbose)
        param_form.addRow(self.cb_keep_paths)
        
        left_layout.addLayout(param_form)
        left_layout.addStretch(1)

        right_layout = QVBoxLayout()

        self.image_combo = QComboBox()
        if os.path.exists(self.image_dir):
            self.image_combo.addItems([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.image_label = QLabel("Sélectionnez une image pour l'afficher")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(600, 400)
        self.image_combo.currentIndexChanged.connect(self.display_image)

        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setMinimumHeight(200)

        right_layout.addWidget(self.image_combo)
        right_layout.addWidget(self.image_label)
        right_layout.addWidget(self.console_output)
        
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.setStyleSheet("""
            QPushButton {
                background-color: #cce5ff;
                border: 1px solid #99ccff;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #99ccff;
            }
        """)

        self.stdout_redirect = EmittingStream()
        self.stdout_redirect.text_written.connect(self.update_console)
        sys.stdout = self.stdout_redirect

        self.display_image()

    def update_form_paths(self, script_type):
        """Met à jour les champs de chemin avec les valeurs par défaut si les champs sont vides."""
        paths = self.default_paths.get(script_type, {})
        if not self.line_path_n.text():
            self.line_path_n.setText(paths.get('path_n', ''))
        if not self.line_path_c.text():
            self.line_path_c.setText(paths.get('path_c', ''))

    def clear_form_paths(self):
        """Efface les champs de chemin du formulaire."""
        self.line_path_n.clear()
        self.line_path_c.clear()

    def display_image(self):
        selected_file = self.image_combo.currentText()
        if selected_file:
            image_path = os.path.join(self.image_dir, selected_file)
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                print(f"Erreur : le fichier {selected_file} n'a pas pu être chargé.")
            else:
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def update_console(self, text):
        self.console_output.insertPlainText(text)
        self.console_output.moveCursor(QTextCursor.End)

    def run_app_script(self):
        app_window = APP.MainWindow()
        app_window.show()
        self.hide()

    def run_atq_script(self):
        self.console_output.clear()
        self.update_form_paths('quality')
        print("\nLancement du script de Training Qualité...\n")
        
        bool_c = self.cb_c.isChecked()
        bool_t = self.cb_t.isChecked()
        path_n = self.line_path_n.text()
        path_c = self.line_path_c.text()
        verbose = self.cb_verbose.isChecked()

        #ATQ.main_quality_of_sleep(bool_c, bool_t, path_n, path_c, verbose=verbose)
        
        if not self.cb_keep_paths.isChecked():
            self.clear_form_paths()
        
    def run_att_script(self):
        self.console_output.clear()
        self.update_form_paths('trouble')
        print("\nLancement du script de Training Trouble...\n")
        
        bool_c = self.cb_c.isChecked()
        bool_t = self.cb_t.isChecked()
        path_n = self.line_path_n.text()
        path_c = self.line_path_c.text()
        verbose = self.cb_verbose.isChecked()

        #ATT.run_ATT(bool_c, bool_t, path_n, path_c, verbose=verbose)

        if not self.cb_keep_paths.isChecked():
            self.clear_form_paths()


def run_menu():
    app = QApplication(sys.argv)
    menu = MainMenu()
    menu.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_menu()

    '''