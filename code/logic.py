import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("code")
from architectures import Model_1

class Logic:
    def __init__(self, console, canvas):
        
        # Console
        self.console = console
        
        # Canvas
        self.canvas = canvas
        
        # Class Attributes
        self.model_architecture = Model_1()
        self.model_name = None
        self.pth = None
    
    
    def loadModel(self, model_name):
        # Load in the model from "/models" directory
        self.pth = self.model_architecture
        self.pth.load_state_dict(torch.load(f"models/{model_name}", weights_only=True))
        self.model_name = model_name
        self.console.add_message(f"Loaded model: {model_name}")
    
    def analyze(self):
        from PySide6.QtGui import QImage
        # Get the Current Canvas Image
        image = self.canvas.canvas.toImage()
        
        # Convert to Grayscale Format.
        image = image.convertToFormat(QImage.Format_Grayscale8)
        
        # Get Width and Height.
        width = image.width()
        height = image.height()
        
        # Get a Pointer to Image Data
        ptr = image.bits()
        arr = np.array(ptr).reshape(height, width)
        
        # Resize Image to 28x28
        image = cv2.resize(arr, (28, 28))
        
        # Flatten image so that it can be used as input
        image = image = torch.tensor(image).view(-1).float()
        
        # Normalize
        image = image / 255.0
        
        # Convert to Tensor Float32
        image = torch.tensor(image, dtype=torch.float32)
        
        self.pth.eval()
        with torch.no_grad():
            # Output
            output = self.pth(image)
            
            # Print the Output Tensor, but as a simple List.
            self.console.add_message(f"Output: {output.tolist()}")
            
            # Find the index of the maximum value
            prediction = torch.argmax(output).item()
            
            # Print Prediction
            self.console.add_message(f"Prediction: {prediction}")