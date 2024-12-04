import sys
from PySide6.QtWidgets import QHBoxLayout, QApplication, QMainWindow, QVBoxLayout, QWidget

# Connect /code/ directory 
sys.path.append("code")
from canvas import DrawingCanvas, CanvasWidget
from options_menu import OptionsMenu
from logic import Logic
from console import ConsoleWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Get Items First
        self.console = ConsoleWindow()
        self.canvas = CanvasWidget()
        self.logic = Logic(self.console, self.canvas.canvas)
        self.options_menu = OptionsMenu(self.logic)
        
        # Load Central Widget
        central_widget = QWidget()
        central_layout = QHBoxLayout()
        
        # Add Widgets in following order:
        # Canvas
        # Options Menu
        # Console
        central_layout.addWidget(self.canvas)
        central_layout.addWidget(self.options_menu)
        central_layout.addWidget(self.console)
        
        # Set the central widget
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
