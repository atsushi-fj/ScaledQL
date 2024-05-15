import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import os

from modified_resnet import ModifiedResNet34,CustomCNN
from utils import hidden_init


class ScaledCriticNet(nn.Module):
    def __init__(self, input_dims, action_dim, name, chkpt_dir, hidden_size=256, n_norm_groups=4):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.name = name
        self.chekcpoint_dir = chkpt_dir
        # self.encoder = ModifiedResNet34(n_norm_groups, input_dims)
        self.encoder = CustomCNN(input_dims)
        fc_input_dims = int(np.prod(self.encoder.encoder_output_dims))
        self.fc1 = nn.Linear(fc_input_dims + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Linear(hidden_size, 1)
        
        self.apply(self._weights_init)
        self.reset_parameters()
        self.to(self.device)
        print(f"Using {self.device} deivce")
        
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.q.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        state = self.encoder(state)
        state = state.view(state.size()[0], -1)
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = F.relu(self.fc2(action_value))
        action_value = F.relu(self.fc3(action_value))
        q = self.q(action_value)
        return q
    
    def save_checkpoint(self, epoch):
        print("... saving checkpoint ...")
        fname = f"{self.name}_{epoch}"
        self.chekcpoint_file = os.path.join(self.chekcpoint_dir, fname)
        torch.save(self.state_dict(), self.chekcpoint_file)
        
    def load_checkpoint(self, epoch):
        print(",,, loading checkpoint ...")
        fname = f"{self.name}_{epoch}"
        self.chekcpoint_file = os.path.join(self.chekcpoint_dir, fname)
        self.load_state_dict(torch.load(self.chekcpoint_file))


class ScaledActorNet(nn.Module):
    def __init__(self, input_dims, output_dims, name, chkpt_dir,
                hidden_size=256, n_norm_groups=4, epsilon=1e-6):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.name = name
        self.chekcpoint_dir = chkpt_dir
        self.epsilon = epsilon
        # self.encoder = ModifiedResNet34(n_norm_groups, input_dims)
        self.encoder = CustomCNN(input_dims)
        fc_input_dims = int(np.prod(self.encoder.encoder_output_dims))
        self.fc1 = nn.Linear(fc_input_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.alpha_head = nn.Linear(hidden_size, output_dims)
        self.beta_head = nn.Linear(hidden_size, output_dims)
                
        self.apply(self._weights_init)
        self.to(self.device)
        print(f"Using {self.device} deivce")
        self.temp_step = 0
    
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, state):
        state = self.encoder(state)
        state = state.view(state.size()[0], -1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        alpha = F.softplus(self.alpha_head(x)) + 1.0
        beta = F.softplus(self.beta_head(x)) + 1.0
        self.temp_step += 1
        return alpha, beta

    def sample(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        x_t = dist.rsample()
        action = x_t * torch.tensor([2., 1., 1.], device=self.device).repeat(x_t.shape[0], 1) + torch.tensor([-1., 0., 0.], device=self.device).repeat(x_t.shape[0], 1)
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log((1 - action.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        action_eval = alpha / (alpha + beta)
        action_eval = action_eval * torch.tensor([2., 1., 1.], device=self.device).repeat(x_t.shape[0], 1) + torch.tensor([-1., 0., 0.], device=self.device).repeat(x_t.shape[0], 1)
        return action, log_prob, action_eval
    
    def save_checkpoint(self, epoch):
        print("... saving checkpoint ...")
        fname = f"{self.name}_{epoch}"
        self.chekcpoint_file = os.path.join(self.chekcpoint_dir, fname)
        torch.save(self.state_dict(), self.chekcpoint_file)
        
    def load_checkpoint(self, epoch):
        print(",,, loading checkpoint ...")
        fname = f"{self.name}_{epoch}"
        self.chekcpoint_file = os.path.join(self.chekcpoint_dir, fname)
        self.load_state_dict(torch.load(self.chekcpoint_file))