# Importing PyTorch and neural network modules
import torch
import torch.nn as nn

# Define a neural network class that inherits from PyTorch's base module
class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        # Initialize the parent nn.Module
        super(Action_Conditioned_FF, self).__init__()
        
        # Input layer: takes 8 input features and outputs 64
        self.fc1 = nn.Linear(8, 64)
        
        # Hidden layer: takes 64 inputs and outputs 32
        self.fc2 = nn.Linear(64, 32)
        
        # Output layer: maps from 32 features to 1 value (binary classification)
        self.output = nn.Linear(32, 1)
        
        # Apply sigmoid to squash output between 0 and 1 (probability)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply ReLU activation on first layer
        x = torch.relu(self.fc1(x))
        
        # Apply ReLU activation on second layer
        x = torch.relu(self.fc2(x))
        
        # Output layer (raw prediction before squashing)
        x = self.output(x)
        
        # Squash the output to a probability
        return self.sigmoid(x)

    def evaluate(self, model, test_loader, loss_function):
        # Set model to evaluation mode (turns off dropout/batch norm etc.)
        model.eval()
        
        # Total loss across all test data
        loss_total = 0.0
        
        # No gradient calculations during evaluation to save memory
        with torch.no_grad():
            for batch in test_loader:
                # Get input features and labels
                inputs = batch['input']
                labels = batch['label']
                
                # Run the model to get predictions
                outputs = model(inputs)
                
                # Calculate loss between predictions and actual labels
                loss = loss_function(outputs, labels)
                
                # Accumulate the loss
                loss_total += loss.item()
        
        # Return average loss over all batches
        return loss_total / len(test_loader)
