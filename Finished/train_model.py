from Data_Loaders import Data_Loaders  # Import the custom data loader class from Data_Loaders.py
from Networks import Action_Conditioned_FF  # Import the neural network model class from Networks.py

import torch  # Import PyTorch library for deep learning operations
import torch.nn as nn  # Import neural network modules from PyTorch
import matplotlib.pyplot as plt  # Import the plotting library to visualize training results


def train_model(no_epochs):  # Define a function called train_model that takes number of training epochs as input

    batch_size =  # You must assign a value here (like 64 or 128). It controls how many samples the model sees per step.
    data_loaders = Data_Loaders(batch_size)  # Create a Data_Loaders object to get train/test data
    model = Action_Conditioned_FF()  # Create an instance of the neural network model


    losses = []  # Initialize a list to keep track of the loss values during training
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)  # Evaluate model on test data before training
    losses.append(min_loss)  # Store the starting loss in the list


    for epoch_i in range(no_epochs):  # Loop through each epoch (full pass through training data)
        model.train()  # Set model to training mode
        for idx, sample in enumerate(data_loaders.train_loader): # Loop through batches of data from training loader
            pass  # Placeholder for training logic (forward pass, loss computation, backpropagation, optimizer step)


if __name__ == '__main__':  # This block runs if the file is executed directly
    no_epochs =  # You must assign a value here (like 10 or 20). It tells how long to train.
    train_model(no_epochs)  # Call the train_model function with the specified number of epochs
