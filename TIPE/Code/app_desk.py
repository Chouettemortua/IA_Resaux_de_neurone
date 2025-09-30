# echo $DISPLAY, if nothing is returned, run the following command in the terminal
# export DISPLAY=:0
# If you are using WSL, run the following command in the terminal   

# Importation des bibliothèques nécessaires
import sys
import os
import pandas as pd
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


if sys.platform == "win32":
    # Pour windows
    os.environ["QT_QPA_PLATFORM"] = "windows"
elif sys.platform == "linux":
    # Pour les systèmes basés sur Linux
    os.environ["QT_QPA_PLATFORM"] = "xcb"
elif sys.platform == "darwin":
    # Pour macOS
    os.environ["QT_QPA_PLATFORM"] = "cocoa"

# Classe pour le formulaire d'ajout d'entrée
class AddEntryFrom(QDockWidget):
    """ Formulaire pour ajouter une entrée utilisateur. """
    def __init__(self, add_entry_callback, correspondance):
        """ Initialisation du formulaire avec les champs nécessaires. 
        Args:
            add_entry_callback (function): Fonction de rappel pour ajouter une entrée."""
        super().__init__()
        self.add_entry_callback = add_entry_callback
        self.correspondance = correspondance

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
        add_row("Genre", gender); row += 1

        bmi = QComboBox()
        bmi.addItems(["Normal", "Surpoid", "Sous-poid", "Obese"])
        add_row("IMC Catégorie", bmi); row += 1

        age = QSpinBox(); age.setRange(0, 120)
        add_row("Age", age); row += 1

        sleep_duration = QDoubleSpinBox(); sleep_duration.setRange(0.0, 24.0); sleep_duration.setSingleStep(1)
        add_row("Durée du sommeil", sleep_duration); row += 1

        physical_activity = QSpinBox(); physical_activity.setRange(0, 1440)
        add_row("Temps d'activité physique", physical_activity); row += 1

        stress = QSpinBox(); stress.setRange(1, 10)
        add_row("Niveau de stress", stress); row += 1

        blood_pressure = QLineEdit()
        blood_pressure.setPlaceholderText("Ex: 120/80")
        add_row("Pression sanguine", blood_pressure); row += 1

        heart_rate = QSpinBox(); heart_rate.setRange(30, 200)
        add_row("BPM", heart_rate); row += 1

        steps = QSpinBox(); steps.setRange(0, 50000); steps.setSingleStep(1000)
        add_row("Pas journalier", steps); row += 1

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
        """ Collecte les données du formulaire et appelle la fonction de rappel. """
        data = {}
        for key, widget in self.fields.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                data[self.correspondance[key]] = widget.value()
            elif isinstance(widget, QComboBox):
                data[self.correspondance[key]] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                data[self.correspondance[key]] = widget.text()
        self.add_entry_callback(data)


# Fenêtre principale de l'application
class MainWindow(QMainWindow):
    """ Fenêtre principale de l'application. """
    def __init__(self):
        """ Initialisation de la fenêtre principale et des composants UI. """

        super().__init__()

        # Configuration de la fenêtre principale
        self.setWindowTitle("Sleep IA")
        self.resize(1000, 600)

        # Initialisation du DataFrame
        self.df = pd.DataFrame()
        self.columns = ["Gender", "Age", "Occupation", "Sleep Duration",
        "Physical Activity Level", "Stress Level", "BMI Category",
        "Blood Pressure", "Heart Rate", "Daily Steps"
        ]
        self.correspondance = {
            "Genre" : self.columns[0],
            "IMC Catégorie" : self.columns[6],
            "Age" : self.columns[1],
            "Durée du sommeil" : self.columns[3],
            "Temps d'activité physique" : self.columns[4],
            "Niveau de stress" : self.columns[5],
            "Pression sanguine" : self.columns[7],
            "BPM" : self.columns[8],
            "Pas journalier" : self.columns[9],
            "Occupation" : self.columns[2],
            "Surpoid" : "overweight",
            "Sous-poid" : "underweight",
            "Obese" : "Obese"
        }
        self.df = pd.DataFrame(columns=self.columns)

        self.init_ui()

    def init_ui(self):
        """ Initialisation de l'interface utilisateur. """

        # Barre d'outils
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)  # toolbar fixe
        self.addToolBar(self.toolbar)

        # Boutons de la barre d'outils
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

        # Splitter pour diviser la table et le formulaire
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Table pour afficher les données
        self.table = QTableWidget()
        self.refresh_table()
        self.splitter.addWidget(self.table)

        # Questionaire
        self.add_entry_form = AddEntryFrom(self.add_entry, self.correspondance)
        self.splitter.addWidget(self.add_entry_form)

        self.setCentralWidget(self.splitter)
        self.splitter.setSizes([int(self.width() * 0.75), int(self.width() * 0.25)])

    def add_entry(self, entry_dict):
        """ Ajoute une entrée au DataFrame et met à jour la table.
        Args:
            entry_dict (dict): Dictionnaire contenant les données de l'entrée. """
        
        # Vérification des colonnes attendues
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
        """ Rafraîchit l'affichage de la table avec les données du DataFrame. """

        # Mise à jour de la table
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
        """ Affiche ou masque le formulaire d'ajout d'entrée.
        Args:
            checked (bool): État du bouton toggle. """
        
        # Afficher ou masquer le formulaire en fonction de l'état du bouton
        if checked:
            self.add_entry_form.show()
            self.splitter.setSizes([int(self.width() * 0.75), int(self.width() * 0.25)])
        else:
            self.add_entry_form.hide()
            self.splitter.setSizes([self.width(), 0])

    def analyse_data(self):
        """ Analyse les données du DataFrame en utilisant les modèles pré-entraînés. """

        # Vérification si le DataFrame est vide
        if self.df.empty:
            QMessageBox.warning(self, "Avertissement", "Aucune donnée à analyser.")
            return
        
        # Chargement des modèles
        file_dir = os.path.abspath(__file__)
        file_dir = os.path.dirname(file_dir)
        model_quality_path = os.path.join(file_dir, 'Saves', 'save_sleep_quality.pkl')
        model_trouble_path = os.path.join(file_dir, 'Saves', 'save_sleep_trouble.pkl')
        model_quality = model_charge(model_quality_path)
        model_trouble = model_charge(model_trouble_path)

        n = min(5, len(self.df))
        recent_entries = self.df.tail(n)

        try:
            # Préparation des données pour le modèle de qualité du sommeil
            df_quality = preprocecing_user(recent_entries)
            """ test de debug
            print("df_quality shape:", df_quality.shape)
            print(df_quality.head())
            """
            # Prédictions de qualité de sommeil
            pred_qualities = [model_quality.predict(row.values)[0] for _, row in df_quality.iterrows()]
            # Filtrer les nan ou valeurs non numériques (au cas où)
            mean_quality = sum(pred_qualities) / len(pred_qualities)

            # Préparation des données pour le modèle de trouble du sommeil
            df_trouble = df_quality.copy() # même preprocessing
            pred_trouble = [model_trouble.predict(row.values)[0] for _, row in df_trouble.iterrows()]
            mean_trouble = sum(pred_trouble) / len(pred_trouble)

            # Mapping des classes de trouble du sommeil
            labels_trouble = {
                0: "Normal (pas de trouble)",
                1: "Apnée du sommeil",
                2: "Insomnie"
            }

            # Calcul de la classe dominante moyenne
            classe_moyenne_trouble = round(mean_trouble)
            label_trouble = labels_trouble.get(classe_moyenne_trouble, "Inconnu")

            # Affichage des résultats
            QMessageBox.information(
                self,
                "Analyse",
                f"Score moyen qualité de sommeil : {mean_quality * 10:.2f}%\n"
                f"Trouble du sommeil détecté : {label_trouble}\n"
            )

        # Gestion des erreurs
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Échec de l'analyse : {str(e)} PS: MARTIN")

    def load_csv(self):
        """ Charge un fichier CSV dans le DataFrame et met à jour la table. """
        # Ouvrir une boîte de dialogue pour sélectionner le fichier CSV
        file, _ = QFileDialog.getOpenFileName(self, "Charger un CSV")
        if file:
            try:
                # Lire le CSV dans un DataFrame temporaire
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

            # Gestion des erreurs
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Échec du chargement : {str(e)}")

    def save_csv(self):
        """ Sauvegarde le DataFrame actuel dans un fichier CSV. """
        # Ouvrir une boîte de dialogue pour sélectionner le fichier de sauvegarde
        file, _ = QFileDialog.getSaveFileName(self, "Sauvegarder CSV")
        if file:
            try:
                self.df.to_csv(file, index=False)
            # Gestion des erreurs
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Échec de la sauvegarde : {str(e)}")

    def clear_table(self):
        """ Vide le DataFrame et la table. """
        self.df = pd.DataFrame(columns=self.columns)
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.refresh_table()


# Lancement de l'application
def run_app():
    """ Lance l'application principale. """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()