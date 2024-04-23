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
        state, action, reward, next_state, done = \
            data[0], data[1], data[2], data[3], data[4]
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)
        return state, action, reward, next_state, done
    