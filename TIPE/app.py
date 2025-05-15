# echo $DISPLAY, if nothing is returned, run the following command in the terminal
# export DISPLAY=:0
# If you are using WSL, run the following command in the terminal   

import sys
import pandas as pd
import csv
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QToolBar, QLabel, QLineEdit,
    QPushButton, QComboBox, QFileDialog, QMessageBox, QFormLayout, QSplitter,
    QDockWidget, QGridLayout, QSizePolicy, QDoubleSpinBox, QSpinBox, QGroupBox
)
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import Qt
from AI_Model import Resaux


class AddEntryFrom(QDockWidget):
    def __init__(self, add_entry_callback):
        super().__init__("Ajouter une entrée")
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

        disorder = QComboBox()
        disorder.addItems(["None", "Insomnia", "Sleep Apnea"])
        add_row("Sleep Disorder", disorder); row += 1

        age = QSpinBox(); age.setRange(0, 120)
        add_row("Age", age); row += 1

        sleep_duration = QDoubleSpinBox(); sleep_duration.setRange(0.0, 24.0); sleep_duration.setSingleStep(1)
        add_row("Sleep Duration", sleep_duration); row += 1

        physical_activity = QSpinBox(); physical_activity.setRange(0, 100)
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

        occupation = QLineEdit()
        occupation.setPlaceholderText("Entrez votre métier")
        add_row("Occupation", occupation); row += 1

        # Bouton d'enregistrement
        self.submit_button = QPushButton("Ajouter à la base")
        self.submit_button.clicked.connect(self.submit_entry)
        self.submit_button.setStyleSheet("padding: 8px; background-color: #cce5ff; border: 1px solid #99ccff;")

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sleep Disorder Tracker")
        self.resize(1000, 600)
        self.df = pd.DataFrame()

        self.init_ui()

    def init_ui(self):
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)  # toolbar fixe
        self.addToolBar(self.toolbar)

        load_action = QAction(QIcon(), "Charger CSV", self)
        load_action.triggered.connect(self.load_csv)
        self.toolbar.addAction(load_action)

        save_action = QAction(QIcon(), "Sauvegarder CSV", self)
        save_action.triggered.connect(self.save_csv)
        self.toolbar.addAction(save_action)

        clear_action = QAction(QIcon(), "Vider la table", self)
        clear_action.triggered.connect(self.clear_table)
        self.toolbar.addAction(clear_action)

        toggle_form_action = QAction(QIcon(), "Afficher/Masquer Formulaire", self)
        toggle_form_action.setCheckable(True)
        toggle_form_action.setChecked(True)
        toggle_form_action.triggered.connect(self.toggle_form_visibility)
        self.toolbar.addAction(toggle_form_action)

        self.columns = [
            "Gender", "BMI Category", "Sleep Disorder", "Age", "Sleep Duration",
            "Physical Activity Level", "Stress Level", "Blood Pressure",
            "Heart Rate", "Daily Steps", "Occupation"
        ]

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self.columns))
        self.table.setHorizontalHeaderLabels(self.columns)
        header = self.table.horizontalHeader()
        font = header.font()
        font.setBold(True)
        header.setFont(font)
        self.splitter.addWidget(self.table)

        self.add_entry_form = AddEntryFrom(self.add_entry)
        self.splitter.addWidget(self.add_entry_form)

        self.setCentralWidget(self.splitter)
        self.splitter.setSizes([int(self.width() * 0.75), int(self.width() * 0.25)])

    def add_entry(self, entry_dict):
        if self.df.empty:
            self.df = pd.DataFrame(columns=entry_dict.keys())
        self.df.loc[len(self.df)] = entry_dict
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

    def load_csv(self):
        file, _ = QFileDialog.getOpenFileName(self, "Charger un CSV")
        if file:
            try:
                self.df = pd.read_csv(file)
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
        self.df = pd.DataFrame()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
