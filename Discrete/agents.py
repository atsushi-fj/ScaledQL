import numpy as np
import torch
import torch.nn.functional as F
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer
from scaled_q_network import ScaledQNetwork, CategoricalScaledQNetwork
import matplotlib.pyplot as plt
import time

class Agent:
    def __init__(self, gamma, epsilon, lr, eps, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir="tmp/dqn"):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.eps = eps
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        
    def choose_action(self, observation):
        raise NotImplementedError
    
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
    
    def sample_memory(self):
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)
                                
        states = torch.tensor(state).to(self.q_eval.device)
        rewards = torch.tensor(reward).to(self.q_eval.device)
        dones = torch.tensor(done).to(self.q_eval.device)
        actions = torch.tensor(action).to(self.q_eval.device)
        states_ = torch.tensor(new_state).to(self.q_eval.device)
        
        return states, actions, rewards, states_, dones
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
        
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min
                        
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()
        
    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
        
    def learn(self):
        raise NotImplementedError
    

class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)
        
        self.q_eval = DeepQNetwork(self.lr, self.eps, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name+"_"+self.algo+"_q_eval",
                                   chkpt_dir=self.chkpt_dir)
        
        self.q_next = DeepQNetwork(self.lr, self.eps, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name+"_"+self.algo+"_q_next",
                                   chkpt_dir=self.chkpt_dir)
        
        print("param", sum(p.numel() for p in self.q_eval.parameters()))
        
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(np.array([observation]),dtype=torch.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()
        
        self.replace_target_network()
        
        states, actions, rewards, states_, dones = self.sample_memory()
        
        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]  # dims -> batch_size x n_actions
        q_next = self.q_next.forward(states_).max(dim=1)[0]
        
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next
        
        loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        
        self.decrement_epsilon()


class ScaledQAgent(Agent):
    def __init__(self, n_norm_groups, dataloader, alpha, v_min, v_max, atoms, *args, **kwargs):
        super(ScaledQAgent, self).__init__(*args, **kwargs)
        self.n_norm_gropus = n_norm_groups
        if dataloader is not None:
            self.dataloader = dataloader
        self.alpha = alpha
        
        self.q_eval = ScaledQNetwork(self.lr, self.eps, self.n_norm_gropus,
                                     self.n_actions, 
                                     input_dims=self.input_dims,
                                     name=self.env_name+"_"+self.algo+"_q_eval",
                                     chkpt_dir=self.chkpt_dir)
        
        self.q_next = ScaledQNetwork(self.lr, self.eps, self.n_norm_gropus,
                                     self.n_actions, 
                                     input_dims=self.input_dims,
                                     name=self.env_name+"_"+self.algo+"_q_next",
                                     chkpt_dir=self.chkpt_dir)
        
        print("param", sum(p.numel() for p in self.q_eval.parameters()))
        self.q_eval.train()
        
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(np.array([observation]),dtype=torch.float).to(self.q_eval.device)
            self.q_eval.eval()
            with torch.no_grad():
                actions = self.q_eval.forward(state)
            action = torch.argmax(actions, dim=1).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def cql_loss(self, q_values, current_action):
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)
        return (logsumexp - q_a).mean()
        
    def learn(self):
        
        self.q_next.eval()
        
        train_loss = 0
        start_epoch_time = time.time()
        for batch, data in enumerate(self.dataloader):
            self.replace_target_network()
            states, actions, rewards, states_, dones = \
                data[0].to(self.q_eval.device), data[1].to(self.q_eval.device), \
                data[2].to(self.q_eval.device), data[3].to(self.q_eval.device), \
                data[4].to(self.q_eval.device)
            q_a_s = self.q_eval(states)
            q_pred = q_a_s.gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                q_next = self.q_next(states_).max(dim=1)[0]
            
            q_next[dones] = 0.0
            q_target = rewards + self.gamma*q_next
            
            reguralizer = self.cql_loss(q_a_s, actions.unsqueeze(1))
            td_error = F.mse_loss(q_pred, q_target.unsqueeze(1))
            self.q_eval.optimizer.zero_grad()
            loss = self.alpha*reguralizer + td_error
            loss.backward()
            self.q_eval.optimizer.step()
            self.learn_step_counter += 1
            if self.learn_step_counter % 10 == 0: print("step: ", self.learn_step_counter)
            if self.learn_step_counter % 100 == 0: self.save_models()
            train_loss += loss.item()
        train_loss /= len(self.dataloader)
        end_epoch_time = time.time()
        print("epoch_time: ", end_epoch_time - start_epoch_time)
        return train_loss
    
    
class ConservativeQAgent(Agent):
    def __init__(self, n_norm_groups, dataloader, alpha, v_min, v_max, atoms, *args, **kwargs):
        super(ConservativeQAgent, self).__init__(*args, **kwargs)
        self.n_norm_gropus = n_norm_groups
        if dataloader is not None:
            self.dataloader = dataloader
        self.alpha = alpha
        self.q_eval = DeepQNetwork(self.lr, self.eps, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name+"_"+self.algo+"_q_eval",
                                   chkpt_dir=self.chkpt_dir)
        
        self.q_next = DeepQNetwork(self.lr, self.eps, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name+"_"+self.algo+"_q_next",
                                   chkpt_dir=self.chkpt_dir)
        
        print("param", sum(p.numel() for p in self.q_eval.parameters()))
    
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(np.array([observation]),dtype=torch.float).to(self.q_eval.device)
            self.q_eval.eval()
            with torch.no_grad():
                actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def cql_loss(self, q_values, current_action):
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)
        return (logsumexp - q_a).mean()
        
    def learn(self):
        
        self.q_eval.train()
        self.q_next.eval()
        
        train_loss = 0
        for batch, data in enumerate(self.dataloader):
            self.replace_target_network()
            states, actions, rewards, states_, dones = \
                data[0].to(self.q_eval.device), data[1].to(self.q_eval.device), \
                data[2].to(self.q_eval.device), data[3].to(self.q_eval.device), \
                data[4].to(self.q_eval.device)
            
            q_a_s = self.q_eval(states)
            q_pred = q_a_s.gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                q_next = self.q_next(states_).max(dim=1)[0]
            
            q_next[dones] = 0.0
            q_target = rewards + self.gamma*q_next
            
            
            reguralizer = self.cql_loss(q_a_s, actions.unsqueeze(1))
            td_error = F.mse_loss(q_pred, q_target.unsqueeze(1))
            loss = self.alpha*reguralizer + td_error
            self.q_eval.optimizer.zero_grad()
            loss.backward()
            self.q_eval.optimizer.step()
            self.learn_step_counter += 1
            if self.learn_step_counter % 10 == 0: print("step: ", self.learn_step_counter)
            if self.learn_step_counter % 100 == 0: self.save_models()
            
            train_loss += loss.item()
        train_loss /= len(self.dataloader)
        return train_loss


class CategoricalScaledQAgent(Agent):
    def __init__(self, n_norm_groups, dataloader, alpha, v_min, v_max, atoms, *args, **kwargs):
        super(CategoricalScaledQAgent, self).__init__(*args, **kwargs)
        self.n_norm_gropus = n_norm_groups
        if dataloader is not None:
            self.dataloader = dataloader
        self.alpha = alpha
        self.q_eval = CategoricalScaledQNetwork(self.lr, self.eps, self.n_norm_gropus,
                                     self.n_actions, 
                                     input_dims=self.input_dims,
                                     name=self.env_name+"_"+self.algo+"_q_eval",
                                     chkpt_dir=self.chkpt_dir, atoms=atoms)
        
        self.q_next = CategoricalScaledQNetwork(self.lr, self.eps, self.n_norm_gropus,
                                     self.n_actions, 
                                     input_dims=self.input_dims,
                                     name=self.env_name+"_"+self.algo+"_q_next",
                                     chkpt_dir=self.chkpt_dir, atoms=atoms)
        
        print("param", sum(p.numel() for p in self.q_eval.parameters()))
        
        self.q_eval.train()
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, atoms).to(self.q_eval.device)
        self.delta = (v_max - v_min) / (atoms - 1)
        
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(np.array([observation]),dtype=torch.float).to(self.q_eval.device)
            self.q_eval.eval()
            with torch.no_grad():
                q_value_probs = self.q_eval.forward(state)
                q_values = (self.support * q_value_probs).sum(dim=-1)
                action = torch.argmax(q_values, dim=1).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def cql_loss(self, q_values, current_action):
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)
        return (logsumexp - q_a).min()
    
    def learn(self):
        
        self.q_next.eval()
        
        train_loss = 0
        for batch, data in enumerate(self.dataloader):
            self.replace_target_network()
            states, actions, rewards, states_, dones = \
                data[0].to(self.q_eval.device), data[1].to(self.q_eval.device), \
                data[2].to(self.q_eval.device), data[3].to(self.q_eval.device), \
                data[4].to(self.q_eval.device)
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1)
            batch_size = states.shape[0]
            
            with torch.no_grad():
                next_q_value_probs = self.q_next(states_)
                next_q_values = (next_q_value_probs * self.support).sum(dim=-1)
                next_actions = next_q_values.argmax(dim=-1)
                next_action_value_probs = next_q_value_probs[range(batch_size), next_actions, :]    
            
            q_value_probs = self.q_eval(states)
            q_values = (self.support * q_value_probs).sum(dim=-1)
            action_value_probs = q_value_probs[range(batch_size), actions, :]
            log_action_value_probs = torch.log(action_value_probs + 1e-6)
                
            
            m = torch.zeros(batch_size * self.atoms).to(self.q_eval.device)
            Tz = rewards + ~dones * self.gamma * self.support.unsqueeze(0)
            Tz.clamp_(min=self.v_min, max=self.v_max)

            b = (Tz - self.v_min) / self.delta
            l, u = b.floor().long(), b.ceil().long()
            
            offset = torch.arange(batch_size).view(-1, 1).to(self.q_eval.device) * self.atoms
            l_idx = (l + offset).flatten() 
            u_idx = (u + offset).flatten()
            
            upper_probs = (next_action_value_probs * (u.float() - b)).flatten() 
            lower_probs = (next_action_value_probs * (b - l.float())).flatten()
            
            m.index_add_(dim=0, index=l_idx, source=upper_probs)
            m.index_add_(dim=0, index=u_idx, source=lower_probs)

            m = m.reshape(batch_size, self.atoms)

            td_error = - (m * log_action_value_probs).sum(dim=-1)
            td_error = torch.mean(td_error)
            
            reguralizer = self.cql_loss(q_values, actions.unsqueeze(1))

            loss = self.alpha*reguralizer + td_error
            self.q_eval.optimizer.zero_grad()
            loss.backward()
            self.q_eval.optimizer.step()
            self.learn_step_counter += 1
            if self.learn_step_counter % 10 == 0: print("step: ", self.learn_step_counter)
            if self.learn_step_counter % 100 == 0: self.save_models()
            
            train_loss += loss.item()
        train_loss /= len(self.dataloader)
        return train_loss
