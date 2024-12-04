import tqdm

# Import Zero Grad
from torch import no_grad


# DataLoaders
from torch.utils.data import DataLoader

# Training Loop #
class TrainingLoop:
    @classmethod
    def dataloader(cls, dataset, batch_size=32, num_workers=4):
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    
    # Training the Model
    @classmethod
    def train(cls, model, optimizer, loss_fn, train_loader, epochs):
        model.train()
        for epoch in range(epochs):
            for images, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                # Flatten Image
                images = images.view(images.size(0), -1).float()
                
                # Zero Parameter Gradients
                optimizer.zero_grad()
                
                # Forward Pass
                output = model(images)
                loss = loss_fn(images, labels)
                
                # Backward Pass
                loss.backward()
                optimizer.step()
    
    
    
    # Testing the Model
    @classmethod
    def evaluate(cls, model, loss_fn, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return test_loss, accuracy