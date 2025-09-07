import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt

# Importation de mes modules
import Core.training.AI_training_Quality as ATQ
import Core.training.AI_training_Trouble as ATT
import UI.app as APP

class MainMenu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Menu Principal")
        self.setGeometry(100, 100, 400, 200)

        # Création du widget central et du layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Titre du menu
        title_label = QLabel("Choisissez un script à lancer :")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title_label)

        # Création des boutons
        self.btn_app = QPushButton("Lancer l'Application UI")
        self.btn_atq = QPushButton("Lancer le script de Training Qualité")
        self.btn_att = QPushButton("Lancer le script de Training Trouble")

        # Connexion des boutons aux fonctions de lancement
        self.btn_app.clicked.connect(self.run_app_script)
        self.btn_atq.clicked.connect(self.run_atq_script)
        self.btn_att.clicked.connect(self.run_att_script)

        # Ajout des boutons au layout
        main_layout.addWidget(self.btn_app)
        main_layout.addWidget(self.btn_atq)
        main_layout.addWidget(self.btn_att)

        # Style pour les boutons
        button_style = """
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
        """
        self.setStyleSheet(button_style)

    def run_app_script(self):
        # Créer une instance de MainWindow et la montrer
        app_window = APP.MainWindow()
        app_window.show()
        # Masquer la fenêtre du menu
        self.hide()

    def run_atq_script(self):
        # Lancer le script de training qualité
        print("\nLancement du script ATQ...\n")
        ATQ.run_ATQ()

    def run_att_script(self):
        # Lancer le script de training trouble
        print("\nLancement du script ATT...\n")
        ATT.run_ATT()

def run_menu():
    app = QApplication(sys.argv)
    menu = MainMenu()
    menu.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_menu()