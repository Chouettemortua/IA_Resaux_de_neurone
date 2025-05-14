# echo $DISPLAY, if nothing is returned, run the following command in the terminal
# export DISPLAY=:0
# If you are using WSL, run the following command in the terminal   

import sys
import re
import csv
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QFileDialog, QTableWidget,
    QTableWidgetItem, QFormLayout, QLineEdit, QLabel, QMessageBox,
    QDockWidget, QComboBox
)
from PyQt6.QtCore import Qt
from AI_Model import Resaux


class AddEntryDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Ajouter une entrée", parent)
        self.setFloating(False)  # Assurer que le dock est ancré
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        self.fields = [
            "Gender", "Age", "Occupation", "Sleep Duration",
            "Physical Activity Level", "Stress Level", "BMI Category",
            "Blood Pressure", "Heart Rate", "Daily Steps", "Sleep Disorder"
        ]
        self.inputs = {}

        content = QWidget()
        main_layout = QVBoxLayout()

        # --- Bouton d'aide en haut
        help_btn = QPushButton("Aide sur les formats")
        help_btn.clicked.connect(self.show_help)
        main_layout.addWidget(help_btn)

        # --- Formulaire avec QComboBox pour certains champs
        form_layout = QFormLayout()

        for field in self.fields:
            if field == "Gender":
                gender_combo = QComboBox()
                gender_combo.addItems(["Male", "Female"])
                form_layout.addRow(field, gender_combo)
                self.inputs[field] = gender_combo
            elif field == "BMI Category":
                bmi_combo = QComboBox()
                bmi_combo.addItems(["Normal", "Overweight", "Underweight", "Obese"])
                form_layout.addRow(field, bmi_combo)
                self.inputs[field] = bmi_combo
            elif field == "Sleep Disorder":
                disorder_combo = QComboBox()
                disorder_combo.addItems(["None", "Insomnia", "Sleep Apnea"])
                form_layout.addRow(field, disorder_combo)
                self.inputs[field] = disorder_combo
            else:
                inp = QLineEdit()
                form_layout.addRow(field, inp)
                self.inputs[field] = inp

        main_layout.addLayout(form_layout)

        # --- Bouton Ajouter en bas
        btn = QPushButton("Ajouter")
        btn.clicked.connect(self.submit_data)
        main_layout.addWidget(btn)

        content.setLayout(main_layout)
        self.setWidget(content)

    def show_help(self):
        # Afficher un message d'aide avec des informations sur le format des champs
        help_message = (
            "Voici les formats attendus :\n\n"
            "Gender: Male ou Female\n"
            "BMI Category: Underweight, Normal, Overweight, Obese\n"
            "Sleep Disorder: None, Insomnia, Sleep Apnea\n"
            "Age: Entier positif\n"
            "Sleep Duration: Nombre à virgule positif (ex. 6.5)\n"
            "Physical Activity Level: Entier positif\n"
            "Stress Level: Entier entre 1 et 10\n"
            "Blood Pressure: Format NNN/NNN (ex. 120/80)\n"
            "Heart Rate: Entier positif\n"
            "Daily Steps: Entier positif\n"
            "Occupation: Toute valeur non vide"
        )
        QMessageBox.information(self, "Aide", help_message)

    def submit_data(self):
        # Récupération des données du formulaire
        values = [self.inputs[field].currentText().strip() if isinstance(self.inputs[field], QComboBox) else self.inputs[field].text().strip() for field in self.fields]

        try:
            gender = values[0]
            if gender not in ["Male", "Female"]:
                raise ValueError("Gender doit être 'Male' ou 'Female'.")

            age = int(values[1])
            if age < 0:
                raise ValueError("Age doit être un entier positif.")

            occupation = values[2]
            if not occupation:
                raise ValueError("Occupation ne peut pas être vide.")

            sleep_duration = float(values[3])
            if sleep_duration <= 0:
                raise ValueError("Sleep Duration doit être un float > 0.")

            activity = int(values[4])
            if activity < 0:
                raise ValueError("Physical Activity Level doit être ≥ 0.")

            stress = int(values[5])
            if not (1 <= stress <= 10):
                raise ValueError("Stress Level doit être entre 1 et 10.")

            bmi = values[6]
            if bmi not in ["Underweight", "Normal", "Overweight", "Obese"]:
                raise ValueError("BMI Category invalide.")

            bp = values[7]
            if not re.match(r'^\d{2,3}/\d{2,3}$', bp):
                raise ValueError("Blood Pressure doit être au format NNN/NNN.")

            hr = int(values[8])
            if hr <= 0:
                raise ValueError("Heart Rate doit être > 0.")

            steps = int(values[9])
            if steps < 0:
                raise ValueError("Daily Steps doit être ≥ 0.")

            disorder = values[10]
            if disorder not in ["None", "Insomnia", "Sleep Apnea"]:
                raise ValueError("Sleep Disorder invalide.")

            self.parent().add_entry_from_values(values)
            self.close()

        except ValueError as e:
            QMessageBox.critical(self, "Erreur de validation", str(e))
            return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IA SLeep")
        self.resize(1000, 700)
        self.sleep_quality = Resaux()
        self.sleep_quality.load("TIPE/Saves/save_sleep_quality.pkl")
        self.sleep_trouble = Resaux()
        self.sleep_trouble.load("TIPE/Saves/save_sleep_trouble.pkl")

        # --- Layout principal ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Menu boutons ---
        menu_layout = QHBoxLayout()
        self.btn_add = QPushButton("Ajouter Entrée")
        self.btn_analyse = QPushButton("Analyser")
        self.btn_import = QPushButton("Importer")
        self.btn_export = QPushButton("Exporter")

        self.btn_add.clicked.connect(self.add_entry)
        self.btn_analyse.clicked.connect(self.analyze_data)
        self.btn_import.clicked.connect(self.import_data)
        self.btn_export.clicked.connect(self.export_data)

        for btn in [self.btn_add, self.btn_analyse, self.btn_import, self.btn_export]:
            menu_layout.addWidget(btn)
        main_layout.addLayout(menu_layout)

        # --- Table des données ---
        self.table = QTableWidget()
        self.columns = [
            "Gender", "Age", "Occupation", "Sleep Duration",
            "Physical Activity Level", "Stress Level", "BMI Category",
            "Blood Pressure", "Heart Rate", "Daily Steps", "Sleep Disorder"
        ]
        self.table.setColumnCount(len(self.columns))
        self.table.setHorizontalHeaderLabels(self.columns)
        main_layout.addWidget(self.table)

        # --- Résultats d'analyse ---
        self.result_label = QLabel("Résultats de l'analyse :")
        main_layout.addWidget(self.result_label)

        # --- Afficher la fenêtre principale après tout ---
        self.show()

    def add_entry(self):
        if hasattr(self, "entry_dock") and self.entry_dock:
            self.entry_dock.close()
        self.entry_dock = AddEntryDock(self)
        self.entry_dock.setFloating(False)  # Ancrer le dock
        self.entry_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.entry_dock)
        self.entry_dock.setFixedWidth(300)
        self.entry_dock.show()

    def add_entry_from_values(self, data):
        # tu peux valider ici aussi si tu veux
        row_pos = self.table.rowCount()
        self.table.insertRow(row_pos)
        for col, val in enumerate(data):
            self.table.setItem(row_pos, col, QTableWidgetItem(val))

    def analyze_data(self):
        count = self.table.rowCount()
        if count == 0:
            QMessageBox.warning(self, "Aucune donnée", "Ajoutez des données avant d'analyser.")
            return

        # Simulation d’analyse IA
        #disorder = self.sleep_trouble.predict()
        #quality = self.sleep_quality.predict()
        disorder = 1
        quality = 1
        # Affichage des résultats
        if disorder == 0 and quality == 0:
            self.result_label.setText(f"Résultats de l'analyse : Trouble du sommeil: non détecté, Qualité du sommeil: faible")
        elif disorder == 1 and quality == 0:
            self.result_label.setText(f"Résultats de l'analyse : Trouble du sommeil: détecté, Qualité du sommeil: faible")
        elif disorder == 1 and quality == 1:
            self.result_label.setText(f"Résultats de l'analyse : Trouble du sommeil: détecté, Qualité du sommeil: bonne")
        else:
            self.result_label.setText(f"Résultats de l'analyse : Trouble du sommeil: non détecté, Qualité du sommeil: bonne")   
        
    def import_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Importer CSV", "", "CSV Files (*.csv)")
        if file_name:
            with open(file_name, newline='') as f:
                reader = csv.reader(f)
                self.table.setRowCount(0)
                for row_data in reader:
                    row_pos = self.table.rowCount()
                    self.table.insertRow(row_pos)
                    for col, val in enumerate(row_data):
                        self.table.setItem(row_pos, col, QTableWidgetItem(val))

    def export_data(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Exporter CSV", "", "CSV Files (*.csv)")
        if file_name:
            with open(file_name, 'w', newline='') as f:
                writer = csv.writer(f)
                for row in range(self.table.rowCount()):
                    row_data = [
                        self.table.item(row, col).text() if self.table.item(row, col) else ""
                        for col in range(self.table.columnCount())
                    ]
                    writer.writerow(row_data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
