import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from modified_resnet import ModifiedResNet18


class ScaledQNetwork(nn.Module):
    def __init__(self, lr, eps, n_norm_groups, n_actions, name, input_dims, chkpt_dir):
        super().__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.encoder = ModifiedResNet18(n_norm_groups)
        encoder_output_dims = self.calculate_encoder_output_dims(input_dims)
        self.learned_spatial_embeddings = nn.Parameter(torch.randn(encoder_output_dims))
        fc_input_dims = int(np.prod(encoder_output_dims))
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=eps)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"Using {self.device} deivce")
        
    def calculate_encoder_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.encoder(state)
        return dims.size()
    
    def forward(self, state):
        encoder = self.encoder(state)
        encoder_state = self.learned_spatial_embeddings * encoder
        encoder_state_ = encoder_state.view(encoder_state.size()[0], -1)
        flat1 = F.relu(self.fc1(encoder_state_))
        actions = self.fc2(flat1)
        return actions
    
    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print(",,, loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))

        
class CategoricalScaledQNetwork(nn.Module):
    def __init__(self, lr, eps, n_norm_groups, n_actions, atoms, name, input_dims, chkpt_dir):
        super().__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.n_actions = n_actions
        self.atoms = atoms
        self.encoder = ModifiedResNet18(n_norm_groups)
        encoder_output_dims = self.calculate_encoder_output_dims(input_dims)
        self.learned_spatial_embeddings = nn.Parameter(torch.randn(encoder_output_dims))
        fc_input_dims = int(np.prod(encoder_output_dims))
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions * atoms)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=eps)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"Using {self.device} deivce")
        
    def calculate_encoder_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.encoder(state)
        return dims.size()
    
    def forward(self, state):
        encoder = self.encoder(state)
        encoder_state = self.learned_spatial_embeddings * encoder
        encoder_state_ = encoder_state.view(encoder_state.size()[0], -1)
        flat1 = F.relu(self.fc1(encoder_state_))
        actions = F.softmax(self.fc2(flat1).view(-1, self.n_actions, self.atoms), dim=-1)
        actions = actions.clamp(min=1e-3) # for avoiding nans
        return actions
    
    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print(",,, loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))
        
