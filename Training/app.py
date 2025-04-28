import sys
from PyQt6.QtWidgets import QApplication, QWidget

# echo $DISPLAY, if nothing is returned, run the following command in the terminal
# export DISPLAY=:0
# If you are using WSL, run the following command in the terminal   

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("IA Sleep")
window.setGeometry(100, 100, 800, 600)

window.show()

sys.exit(app.exec())

