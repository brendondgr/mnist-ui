# Import General
import tqdm
import os

# Import Zero Grad
from torch import no_grad, max, save

# DataLoaders
from torch.utils.data import DataLoader

# Training Loop #
class TrainingLoop:
    @classmethod
    def dataloader(cls, dataset, batch_size=32, num_workers=4):
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    
    # Training the Model
    @classmethod
    def train_model(cls, model, optimizer, loss_fn, train_loader, epochs):
        # Set Model to Train
        model.train()
        
        # Loss
        temp_loss = 0.0
        
        # Training Loop
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Previous Loss: {temp_loss if temp_loss != 0.0 else 'N/A'}]"):
                # Flatten Image
                images = images.view(images.size(0), -1).float()
                
                # Normalize images
                images = images / 255.0
                
                # Zero Parameter Gradients
                optimizer.zero_grad()
                
                # Forward Pass
                output = model(images)
                loss = loss_fn(output, labels)
                
                # Backward Pass
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # print(labels)
                # print(output)
                
                # Check if the max in the final column's index is not equal to labels
                _, predicted = max(output.data, 1)
                correct = (predicted == labels).sum().item()
            
            # Calculate temp_loss
            temp_loss = f"{(running_loss / len(train_loader)):.4f}"
        
        # Return Model
        return model
    
    
    
    # Testing the Model
    @classmethod
    def evaluate(cls, model, loss_fn, test_loader):
        # Set model for evaluation
        model.eval()
        
        # Variables
        test_loss = 0
        correct = 0
        total=0
        
        # Begin
        with no_grad():
            for images, labels in test_loader:
                # Get Images
                images = images.view(images.size(0), -1).float()
                
                # Outputs
                outputs = model(images)
                
                # Loss
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Return Results
        print(f"Test Loss: {(test_loss / len(test_loader)):.4f}")
        print(f"Accuracy: {(100 * correct / total):.2f}%")
    
    @classmethod
    def save_model(cls, model, path):
        # Create Folder if not exist
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        
        # Name
        number = int(len(os.listdir(path))) + 1
        name = f"{path}/model_{number}.pth"
        
        save(model.state_dict(), name)
        
        # Print
        print(f"Model Saved: {name}")