from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch
import pickle

# Custom Dataset to load CSV and apply normalization
class CustomDataset(Dataset):
    def __init__(self, csv_file, normalizer_path):
        # Read CSV into a pandas DataFrame
        df = pd.read_csv(csv_file)
        
        # Drop the target column 'collision' to get features (X)
        self.X = df.drop(columns='collision').values.astype('float32')
        
        # Extract the label column 'collision'
        self.y = df['collision'].values.astype('float32').reshape(-1, 1)
        
        # Load the scaler object from file to normalize features
        with open(normalizer_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Normalize the features using the loaded scaler
        self.X = scaler.transform(self.X)

    def __len__(self):
        # Returns total number of samples
        return len(self.X)

    def __getitem__(self, idx):
        # Returns a sample as a dictionary with input and label
        return {
            'input': torch.tensor(self.X[idx]),
            'label': torch.tensor(self.y[idx])
        }

# Class to handle training and testing DataLoaders
class Data_Loaders:
    def __init__(self, batch_size):
        # Load the full dataset
        full_dataset = CustomDataset('training_data.csv', 'normalizer.pkl')
        
        # Split 80% for training, 20% for testing
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_set, test_set = random_split(full_dataset, [train_size, test_size])
        
        # Create batched and shuffled DataLoaders
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
