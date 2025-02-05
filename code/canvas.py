from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np

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

    
    def get_canvas_as_array(self):
        """
        Converts the canvas (QPixmap) into a NumPy array.

        Returns:
            numpy.ndarray: A 3D array representing the image in RGB format.
        """
        # Convert QPixmap to QImage
        qimage = self.canvas.toImage()

        # Ensure the QImage format is RGB32 for compatibility
        qimage = qimage.convertToFormat(QtGui.QImage.Format_RGB32)

        # Extract image dimensions
        width = qimage.width()
        height = qimage.height()

        # Access the raw pixel data as a memoryview
        ptr = qimage.bits()

        # Convert the memoryview to a NumPy array
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))  # RGBA format

        # Remove the alpha channel if not needed
        arr = arr[:, :, :3]  # Keep only RGB channels
        
        # Ignore RGB Channels, Make it 1x280x280...
        arr = arr.mean(axis=2)  # Convert to grayscale by averaging RGB channels
        
        # Change Shape from 1x280x280 to 28x28. Skip 10 pixels
        arr = arr[::10, ::10]
        
        # Unsqueeze to add a channel dimension
        arr = arr[np.newaxis, :, :]
        
        # Convert each element of array to float32
        arr = arr.astype(np.float32)
        
        # Normalize
        arr = arr / 255.0

        return arr

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