import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget
from PySide6.QtCore import QTime

class ConsoleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Console")
        self.setGeometry(100, 100, 600, 400)

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def add_message(self, message):
        current_time = QTime.currentTime().toString("hh:mm:ss")
        self.text_edit.append(f"[{current_time}] {message}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConsoleWindow()
    window.show()
    window.add_message("Console started.")
    sys.exit(app.exec())