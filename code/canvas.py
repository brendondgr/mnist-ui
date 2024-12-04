from PySide6 import QtCore, QtGui, QtWidgets

class DrawingCanvas(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 280)  # Set the fixed size of the canvas
        self.canvas = QtGui.QPixmap(self.size())
        self.canvas.fill(QtGui.QColor("white"))
        self.setPixmap(self.canvas)
        self.drawing = False
        self.brush_size = 10

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            self.draw_point(event.pos(), QtCore.Qt.black)
        elif event.button() == QtCore.Qt.RightButton:
            self.drawing = True
            self.last_point = event.pos()
            self.draw_point(event.pos(), QtCore.Qt.white)

    def mouseMoveEvent(self, event):
        if self.drawing:
            color = QtCore.Qt.black if event.buttons() & QtCore.Qt.LeftButton else QtCore.Qt.white
            self.draw_line(self.last_point, event.pos(), color)
            self.last_point = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() in [QtCore.Qt.LeftButton, QtCore.Qt.RightButton]:
            self.drawing = False

    def draw_point(self, point, color):
        painter = QtGui.QPainter(self.canvas)
        painter.setPen(QtGui.QPen(color, self.brush_size))
        painter.drawPoint(point)
        painter.end()
        self.setPixmap(self.canvas)

    def draw_line(self, start_point, end_point, color):
        painter = QtGui.QPainter(self.canvas)
        painter.setPen(QtGui.QPen(color, self.brush_size))
        painter.drawLine(start_point, end_point)
        painter.end()
        self.setPixmap(self.canvas)

    def clear_canvas(self):
        self.canvas.fill(QtGui.QColor("white"))
        self.setPixmap(self.canvas)

class CanvasWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        # Create layout
        layout = QtWidgets.QVBoxLayout()
        
        # Create canvas
        self.canvas = DrawingCanvas()
        
        # Create clear button
        clear_button = QtWidgets.QPushButton("Clear")
        clear_button.clicked.connect(self.canvas.clear_canvas)
        
        # Create slider for brush size
        brush_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        brush_size_slider.setRange(0, 25)
        brush_size_slider.setValue(10)
        
        # Create label to display brush size value
        brush_size_label = QtWidgets.QLabel("10")
        
        # Connect slider value change to update brush size and label
        brush_size_slider.valueChanged.connect(lambda value: (
            setattr(self.canvas, 'brush_size', value),
            brush_size_label.setText(str(value))
        ))
        
        # Add widgets to layout
        layout.addWidget(self.canvas)
        
        controls_layout = QtWidgets.QHBoxLayout()
        
        controls_layout.addWidget(clear_button)
        controls_layout.addWidget(brush_size_slider)
        controls_layout.addWidget(brush_size_label)
        
        layout.addLayout(controls_layout)
        
        # Set the main layout
        self.setLayout(layout)