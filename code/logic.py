import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("code")
from architectures import *

class Logic:
    def __init__(self, console, canvas):
        
        # Console
        self.console = console
        
        # Canvas
        self.canvas = canvas
        
        # Class Attributes
        self.model_architecture = MediumModel()
        self.model_name = None
        self.pth = None
    
    
    def loadModel(self, model_name):
        # Load in the model from "/models" directory
        self.pth = self.model_architecture
        self.pth.load_state_dict(torch.load(
            f"models/{model_name}",
            map_location='cpu',  # Force-load to CPU
            weights_only=True
        ))
        self.model_name = model_name
        self.console.add_message(f"Loaded model: {model_name}")
    
    def save_canvas_to_csv(self, canvas):
        # canvas is a numpy array (1x28x28)
        # Save to CSV
        np.savetxt("canvas.csv", canvas, delimiter=",")
    
    def invert_colors(self, array):
        # Take Any 1 Values and Convert them to 0s, and vice versa
        array[array == 1] = 2
        array[array == 0] = 1
        array[array == 2] = 0
        
        return array
    
    def analyze(self):
        # Get array
        array = self.invert_colors(self.canvas.get_canvas_as_array())

        # squeeze
        new_array = np.squeeze(array)
        self.save_canvas_to_csv(new_array)
        
        # Preprocess the image
        tensor_image = torch.tensor(array).unsqueeze(0)

        self.pth.eval()
        with torch.no_grad():
            output = self.pth(tensor_image)

            # Output results
            # self.console.add_message(f"Output: {output.tolist()}")
            prediction = torch.argmax(output).item()
            self.console.add_message(f"Prediction: {prediction}")