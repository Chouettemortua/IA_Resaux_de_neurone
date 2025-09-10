# echo $DISPLAY, if nothing is returned, run the following command in the terminal
# export DISPLAY=:0
# If you are using WSL, run the following command in the terminal   

# Importation des bibliothèques nécessaires
import sys
import os
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QTableWidget, QTableWidgetItem, QToolBar, QLabel, QLineEdit,
    QPushButton, QComboBox, QFileDialog, QMessageBox, QSplitter,
    QDockWidget, QGridLayout, QSizePolicy, QDoubleSpinBox, QSpinBox, QGroupBox
)
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import Qt

# Ajout du chemin pour l'importation des modules personnalisés
from Core.training.training_utils import model_charge

from Core.preprocessing.preprocessing import preprocecing_user


# Configuration de l'environnement pour éviter les erreurs de plateforme
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Classe pour le formulaire d'ajout d'entrée
class AddEntryFrom(QDockWidget):
    def __init__(self, add_entry_callback):
        super().__init__()
        self.add_entry_callback = add_entry_callback

        self.form_widget = QWidget()
        self.setWidget(self.form_widget)

        main_layout = QVBoxLayout()
        grid = QGridLayout()

        self.fields = {}

        row = 0

        def add_row(label_text, widget):
            label = QLabel(label_text)
            grid.addWidget(label, row, 0)
            grid.addWidget(widget, row, 1)
            self.fields[label_text] = widget

        # Champs avec widgets appropriés
        gender = QComboBox()
        gender.addItems(["Male", "Female"])
        add_row("Gender", gender); row += 1

        bmi = QComboBox()
        bmi.addItems(["Normal", "Overweight", "Underweight", "Obese"])
        add_row("BMI Category", bmi); row += 1

        age = QSpinBox(); age.setRange(0, 120)
        add_row("Age", age); row += 1

        sleep_duration = QDoubleSpinBox(); sleep_duration.setRange(0.0, 24.0); sleep_duration.setSingleStep(1)
        add_row("Sleep Duration", sleep_duration); row += 1

        physical_activity = QSpinBox(); physical_activity.setRange(0, 1440)
        add_row("Physical Activity Level", physical_activity); row += 1

        stress = QSpinBox(); stress.setRange(1, 10)
        add_row("Stress Level", stress); row += 1

        blood_pressure = QLineEdit()
        blood_pressure.setPlaceholderText("Ex: 120/80")
        add_row("Blood Pressure", blood_pressure); row += 1

        heart_rate = QSpinBox(); heart_rate.setRange(30, 200)
        add_row("Heart Rate", heart_rate); row += 1

        steps = QSpinBox(); steps.setRange(0, 50000); steps.setSingleStep(1000)
        add_row("Daily Steps", steps); row += 1

        occupation = QComboBox()
        occupation.addItems(['working', 'unemployed', 'student', 'retired', 'other'])
        add_row("Occupation", occupation); row += 1

        # Bouton d'enregistrement
        self.submit_button = QPushButton("Ajouter à la base")
        self.submit_button.clicked.connect(self.submit_entry)

        # Group box pour regrouper proprement le formulaire
        group_box = QGroupBox("Formulaire d'entrée utilisateur")
        group_box.setLayout(grid)
        group_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        main_layout.addWidget(group_box)
        main_layout.addWidget(self.submit_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.form_widget.setLayout(main_layout)

        # Style général
        self.setStyleSheet("""
            QLabel {
                font-weight: bold;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 1px solid lightgray;
                padding: 8px;
                margin-top: 10px;
            }
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

    def submit_entry(self):
        data = {}
        for key, widget in self.fields.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                data[key] = widget.value()
            elif isinstance(widget, QComboBox):
                data[key] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                data[key] = widget.text()
        self.add_entry_callback(data)


# Fenêtre principale de l'application
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sleep IA")
        self.resize(1000, 600)
        self.df = pd.DataFrame()
        self.columns = ["Gender", "Age", "Occupation", "Sleep Duration",
        "Physical Activity Level", "Stress Level", "BMI Category",
        "Blood Pressure", "Heart Rate", "Daily Steps"
        ]
        self.df = pd.DataFrame(columns=self.columns)

        self.init_ui()

    def init_ui(self):
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)  # toolbar fixe
        self.addToolBar(self.toolbar)

        analyse_action = QAction(QIcon(), "Analyser", self)
        analyse_action.triggered.connect(self.analyse_data)
        self.toolbar.addAction(analyse_action)
        self.toolbar.addSeparator()

        load_action = QAction(QIcon(), "Charger CSV", self)
        load_action.triggered.connect(self.load_csv)
        self.toolbar.addAction(load_action)
        self.toolbar.addSeparator()

        save_action = QAction(QIcon(), "Sauvegarder CSV", self)
        save_action.triggered.connect(self.save_csv)
        self.toolbar.addAction(save_action)
        self.toolbar.addSeparator()

        clear_action = QAction(QIcon(), "Vider la table", self)
        clear_action.triggered.connect(self.clear_table)
        self.toolbar.addAction(clear_action)
        self.toolbar.addSeparator()

        toggle_form_action = QAction(QIcon(), "Afficher/Masquer Formulaire", self)
        toggle_form_action.setCheckable(True)
        toggle_form_action.setChecked(True)
        toggle_form_action.triggered.connect(self.toggle_form_visibility)
        self.toolbar.addAction(toggle_form_action)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.table = QTableWidget()
        self.refresh_table()
        self.splitter.addWidget(self.table)

        self.add_entry_form = AddEntryFrom(self.add_entry)
        self.splitter.addWidget(self.add_entry_form)

        self.setCentralWidget(self.splitter)
        self.splitter.setSizes([int(self.width() * 0.75), int(self.width() * 0.25)])

    def add_entry(self, entry_dict):
        expected_columns = self.columns

        # Créer une nouvelle ligne conforme à l'ordre attendu
        new_row = {col: entry_dict.get(col, "") for col in expected_columns}

        # Initialiser le DataFrame avec les bonnes colonnes si vide
        if self.df.empty:
            self.df = pd.DataFrame(columns=expected_columns)

        # Ajouter la nouvelle ligne
        self.df.loc[len(self.df)] = new_row
        self.refresh_table()

    def refresh_table(self):
        self.table.setRowCount(len(self.df))
        self.table.setColumnCount(len(self.df.columns))
        self.table.setHorizontalHeaderLabels(self.df.columns.tolist())

        for row_idx, row in self.df.iterrows():
            for col_idx, val in enumerate(row):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(val)))

        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        self.table.setShowGrid(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        header = self.table.horizontalHeader()
        font = header.font()
        font.setBold(True)
        header.setFont(font)

    def toggle_form_visibility(self, checked):
        if checked:
            self.add_entry_form.show()
            self.splitter.setSizes([int(self.width() * 0.75), int(self.width() * 0.25)])
        else:
            self.add_entry_form.hide()
            self.splitter.setSizes([self.width(), 0])

    def analyse_data(self):
        if self.df.empty:
            QMessageBox.warning(self, "Avertissement", "Aucune donnée à analyser.")
            return
        
        # Chargement des modèles
        model_quality = model_charge("TIPE/Code/Saves/save_sleep_quality.pkl")
        model_trouble = model_charge("TIPE/Code/Saves/save_sleep_trouble.pkl")

        n = min(5, len(self.df))
        recent_entries = self.df.tail(n)

        try:
            # Préparation des données pour le modèle de qualité du sommeil
            df_quality = preprocecing_user(recent_entries)
            #print("df_quality shape:", df_quality.shape)
            #print(df_quality.head())
            pred_qualities = [model_quality.predict(row.values)[0] for _, row in df_quality.iterrows()]
            # Filtrer les nan ou valeurs non numériques
            mean_quality = sum(pred_qualities) / len(pred_qualities)

            # Préparation des données pour le modèle de trouble du sommeil
            df_trouble = df_quality.copy()
            pred_trouble = [model_trouble.predict(row.values)[0] for _, row in df_trouble.iterrows()]
            mean_trouble = sum(pred_trouble) / len(pred_trouble)

            # Mapping des classes de trouble du sommeil
            labels_trouble = {
                1: "Normal (pas de trouble)",
                2: "Apnée du sommeil",
                3: "Insomnie"
            }

            # Calcul de la classe dominante moyenne
            classe_moyenne_trouble = round(mean_trouble)
            label_trouble = labels_trouble.get(classe_moyenne_trouble, "Inconnu")

            QMessageBox.information(
                self,
                "Analyse",
                f"Score moyen qualité de sommeil : {mean_quality * 100:.2f}%\n"
                f"Trouble du sommeil détecté : {label_trouble}\n"
            )

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Échec de l'analyse : {str(e)}")
       
    def load_csv(self):
        file, _ = QFileDialog.getOpenFileName(self, "Charger un CSV")
        if file:
            try:
                temp_df = pd.read_csv(file)

                # Vérification stricte des colonnes attendues
                expected_columns = set(self.columns)

                file_columns = set(temp_df.columns)

                missing = expected_columns - file_columns
                extra = file_columns - expected_columns

                if missing or extra:
                    raise ValueError(f"Colonnes incorrectes.\nManquantes : {missing}\nInattendues : {extra}")

                # Si toutes les colonnes sont valides, charger
                self.df = temp_df[self.columns]  # pour garder un ordre cohérent
                self.refresh_table()

            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Échec du chargement : {str(e)}")

    def save_csv(self):
        file, _ = QFileDialog.getSaveFileName(self, "Sauvegarder CSV")
        if file:
            try:
                self.df.to_csv(file, index=False)
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Échec de la sauvegarde : {str(e)}")

    def clear_table(self):
        self.df = pd.DataFrame(columns=self.columns)
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.refresh_table()


# Lancement de l'application
def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
