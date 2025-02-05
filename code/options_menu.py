import sys
import os
from PySide6.QtWidgets import QVBoxLayout, QApplication, QWidget, QComboBox, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt

class OptionsMenu(QWidget):
    def __init__(self, logic):
        super().__init__()
        
        # Logic
        self.logic = logic
        
        # Overall Layout
        layout = QVBoxLayout()

        # ROW 1
        row1 = QWidget()
        row1_layout = QHBoxLayout()
        
        # Create QCombo
        self.dropdown = QComboBox()
        self.dropdown.addItems(os.listdir("models"))
        self.dropdown.setCurrentIndex(0)
        
        # Create Load Button
        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.loadModel)
        
        # Add Row1 Items
        row1_layout.addWidget(self.dropdown)
        row1_layout.addWidget(load_button)
        
        # Row 2
        row2 = QWidget()
        row2_layout = QHBoxLayout()
        
        # ROW 2
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze)
        row2_layout.addWidget(self.analyze_button)

        # Add Rows
        row1.setLayout(row1_layout)
        layout.addWidget(row1)
        row2.setLayout(row2_layout)
        layout.addWidget(row2)
        
        # Set the layout for the widget
        self.setLayout(layout)
        
        # Make Layout Qt.AlignTop
        layout.setAlignment(Qt.AlignTop)

    def loadModel(self):
        # Placeholder function for loading a model
        self.logic.loadModel(self.dropdown.currentText())
        
    def analyze(self):
        # Run the Logic for Analyzing Current Canvas
        if self.logic.pth is None:
            self.logic.console.add_message("No model loaded.")
        else:
            self.logic.analyze()