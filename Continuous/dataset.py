import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class OfflineDataset(Dataset):
    def __init__(self, data_dir, log_file):
        self.data_dir = data_dir
        self.df = pd.read_csv(log_file)
        
    def __len__(self):
        return int(self.df.tail(1)["n_save_data"])
    
    def __getitem__(self, idx):
        data_path = f"{self.data_dir}/{idx}.npy"
        data = np.load(data_path, allow_pickle=True)
        state, action, reward, next_state, mask = \
            data[0], data[1], data[2], data[3], data[4]
            
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor([reward])
        mask = torch.FloatTensor([mask])
        return state, action, reward, next_state, mask
