import math
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import copy

from sac_networks import CriticNet, ActorNet
from scaled_networks import ScaledCriticNet, ScaledActorNet
from buffer import ReplayMemory
from utils import soft_update, hard_update, convert_network_grad_to_false


class Agent:
    def __init__(self, input_dims, action_dim, auto_alpha, alpha,
                 chkpt_dir, lr_actor, lr_critic, gamma, hidden_size, tau, batch_size):
        self.tau = tau
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.chkpt_dir = chkpt_dir
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using ", self.device)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if not evaluate:
            action, _, _ = self.actor_net.sample(state)
        else:
            _, _, action = self.actor_net.sample(state) 
        return action.cpu().detach().numpy().reshape(-1)
    
    def save_models(self):
        self.actor_net.save_checkpoint()
        self.critic1_net.save_checkpoint()
        self.critic2_net.save_checkpoint()
        self.critic1_net_target.save_checkpoint()
        self.critic2_net_target.save_checkpoint()
        
    def load_models(self):
        self.actor_net.load_checkpoint()
        self.critic1_net.load_checkpoint()
        self.critic2_net.load_checkpoint()
        self.critic1_net_target.load_checkpoint()
        self.critic2_net_target.load_checkpoint()

        
class SACAgent(Agent):
    def __init__(self, memory_size, *args, **kwargs):
        super(SACAgent, self).__init__(*args, **kwargs)
        self.memory = ReplayMemory(memory_size)
        
        self.actor_net = ActorNet(input_dims=self.input_dims, output_dims=self.action_dim, 
                                  name="actor", chkpt_dir=self.chkpt_dir, hidden_size=self.hidden_size)
        self.critic1_net = CriticNet(input_dims=self.input_dims, action_dim=self.action_dim,
                                           name="critic1", chkpt_dir=self.chkpt_dir,
                                           hidden_size=self.hidden_size)
        self.critic2_net = CriticNet(input_dims=self.input_dims, action_dim=self.action_dim, 
                                           name="critic2", chkpt_dir=self.chkpt_dir,
                                           hidden_size=self.hidden_size)
        self.critic1_net_target = CriticNet(input_dims=self.input_dims, action_dim=self.action_dim, 
                                           name="critic1_target", chkpt_dir=self.chkpt_dir,
                                           hidden_size=self.hidden_size)
        self.critic2_net_target = CriticNet(input_dims=self.input_dims, action_dim=self.action_dim, 
                                           name="critic2_target", chkpt_dir=self.chkpt_dir,
                                           hidden_size=self.hidden_size)
        
        hard_update(self.critic1_net_target, self.critic1_net)
        hard_update(self.critic2_net_target, self.critic2_net)

        convert_network_grad_to_false(self.critic1_net_target)
        convert_network_grad_to_false(self.critic2_net_target)
        
        self.actor_optim = optim.Adam(self.actor_net.parameters(), self.lr_actor)
        self.critic_optim = optim.Adam(list(self.critic1_net.parameters()) + list(self.critic2_net.parameters()), self.lr_critic)

        self.target_entropy = -torch.prod(torch.Tensor(self.action_dim).to(self.device)).item()
        
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr_critic)
        
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=self.lr_critic) 
        
        self.step = 0
        self.critic_loss_save = 0
        self.actor_loss_save = 0
    
    def update_parameters(self):
        self.step += 1

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size=self.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor_net.sample(next_state_batch)
            next_q1_values_target = self.critic1_net_target(next_state_batch, next_action)
            next_q2_values_target = self.critic2_net_target(next_state_batch, next_action)
            next_q_values_target = torch.min(next_q1_values_target, next_q2_values_target) - self.alpha * next_log_pi
            next_q_values = reward_batch + mask_batch * self.gamma * next_q_values_target

        q1_values = self.critic1_net(state_batch, action_batch)
        q2_values = self.critic2_net(state_batch, action_batch)
        critic1_loss = F.mse_loss(q1_values, next_q_values)
        critic2_loss = F.mse_loss(q2_values, next_q_values)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        action, log_pi, _ = self.actor_net.sample(state_batch)
        q1_values = self.critic1_net(state_batch, action)
        q2_values = self.critic2_net(state_batch, action)
        q_values = torch.min(q1_values, q2_values)

        actor_loss = ((self.alpha * log_pi) - q_values).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        soft_update(self.critic1_net_target, self.critic1_net, self.tau)
        soft_update(self.critic2_net_target, self.critic2_net, self.tau)

        self.critic_loss_save += critic_loss
        self.actor_loss_save += actor_loss
        if self.step % 1000 == 0:
            self.critic_loss_save /= 1000
            self.actor_loss_save /= 1000
            print("loss", self.critic_loss_save.item(), self.actor_loss_save.item())
            print("alpha", self.alpha)
            self.critic_loss_save = 0
            self.actor_loss_save = 0
            
        return critic_loss.item(), actor_loss.item()
        

class ScaledQLAgent(Agent):
    def __init__(self, n_norm_groups, with_lagrange, cql_weight, target_action_gap, temparature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_lagrange = with_lagrange
        self.cql_weight = cql_weight
        self.target_action_gap = target_action_gap
        self.temp = temparature
        
        self.actor_net = ScaledActorNet(input_dims=self.input_dims, output_dims=self.action_dim, 
                                        name="actor", chkpt_dir=self.chkpt_dir,
                                        n_norm_groups=n_norm_groups, hidden_size=self.hidden_size)
        self.critic1_net = ScaledCriticNet(input_dims=self.input_dims, action_dim=self.action_dim,
                                           name="critic1", chkpt_dir=self.chkpt_dir,
                                           n_norm_groups=n_norm_groups, hidden_size=self.hidden_size)
        self.critic2_net = ScaledCriticNet(input_dims=self.input_dims, action_dim=self.action_dim, 
                                           name="critic2", chkpt_dir=self.chkpt_dir,
                                           n_norm_groups=n_norm_groups, hidden_size=self.hidden_size)
        self.critic1_net_target = ScaledCriticNet(input_dims=self.input_dims, action_dim=self.action_dim, 
                                           name="critic1_target", chkpt_dir=self.chkpt_dir,
                                           n_norm_groups=n_norm_groups, hidden_size=self.hidden_size)
        self.critic2_net_target = ScaledCriticNet(input_dims=self.input_dims, action_dim=self.action_dim, 
                                           name="critic2_target", chkpt_dir=self.chkpt_dir,
                                           n_norm_groups=n_norm_groups, hidden_size=self.hidden_size)

        hard_update(self.critic1_net_target, self.critic1_net)
        hard_update(self.critic2_net_target, self.critic2_net)

        convert_network_grad_to_false(self.critic1_net_target)
        convert_network_grad_to_false(self.critic2_net_target)
        
        self.actor_optim = optim.Adam(self.actor_net.parameters(), self.lr_actor)
        self.critic_optim = optim.Adam(list(self.critic1_net.parameters()) + list(self.critic2_net.parameters()), self.lr_critic)

        self.target_entropy = -torch.prod(torch.Tensor(self.action_dim).to(self.device)).item()
        
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr_critic)
        
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=self.lr_critic) 
        
        self.step = 0
        self.critic_loss_save = 0
        self.actor_loss_save = 0
    
    def calc_policy_loss(self, state_batch, alpha):
        action, log_pi, _ = self.actor_net.sample(state_batch)
        q1_values = self.critic1_net(state_batch, action)
        q2_values = self.critic2_net(state_batch, action)
        q_values = torch.min(q1_values, q2_values)
        actor_loss = ((alpha * log_pi) - q_values).mean()
        return actor_loss, log_pi
        
    def _compute_policy_values(self, obs_pi, obs_q):
        actions_pred, log_pis, _ = self.actor_net.sample(obs_pi)
        qs1 = self.critic1_net(obs_q, actions_pred)
        qs2 = self.critic2_net(obs_q, actions_pred)
        return qs1 - log_pis, qs2 - log_pis
    
    def _compute_random_values(self, obs, actions, critic):
        random_values = critic(obs, actions)
        random_log_probs = math.log(0.5 ** self.action_dim)
        return random_values - random_log_probs

    def update_parameters(self, data):
        self.step += 1
        
        states, actions, rewards, next_states, masks = data[0], data[1],\
                                                  data[2], data[3], data[4]
                                                                                
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        masks = masks.to(self.device)
        
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pi = self.calc_policy_loss(states, current_alpha)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha.exp() * (log_pi.cpu() + self.target_entropy).detach().cpu()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor_net.sample(next_states)
            next_q1_values_target = self.critic1_net_target(next_states, next_action)
            next_q2_values_target = self.critic2_net_target(next_states, next_action)
            next_q_values_target = torch.min(next_q1_values_target, next_q2_values_target) - self.alpha * next_log_pi
            next_q_values = rewards + masks * self.gamma * next_q_values_target
            
        q1_values = self.critic1_net(states, actions)
        q2_values = self.critic2_net(states, actions)
        
        critic1_loss = F.mse_loss(q1_values, next_q_values)
        critic2_loss = F.mse_loss(q2_values, next_q_values)
        
        # CQL   
        random_actions = torch.FloatTensor(q1_values.shape[0] * 1, actions.shape[-1]).uniform_(-1, 1).to(self.device)
        num_repeat = int(random_actions.shape[0] / states.shape[0])
        temp_states = states.unsqueeze(1).repeat(1, num_repeat, 1, 1, 1).view(states.shape[0] * num_repeat, states.shape[-3], states.shape[-2], states.shape[-1])
        temp_next_states = next_states.unsqueeze(1).repeat(1, num_repeat, 1, 1, 1).view(next_states.shape[0] * num_repeat, next_states.shape[-3], next_states.shape[-2], next_states.shape[-1])
        
        current_pi_values1, current_pi_values2 = self._compute_policy_values(temp_states, temp_states)
        next_pi_values1, next_pi_values2 = self._compute_policy_values(temp_next_states, temp_states)
        
        random_values1 = self._compute_random_values(temp_states, random_actions, self.critic1_net).reshape(states.shape[0], num_repeat, 1)
        random_values2 = self._compute_random_values(temp_states, random_actions, self.critic2_net).reshape(states.shape[0], num_repeat, 1)
        
        current_pi_values1 = current_pi_values1.reshape(states.shape[0], num_repeat, 1)
        current_pi_values2 = current_pi_values2.reshape(states.shape[0], num_repeat, 1)
        
        next_pi_values1 = next_pi_values1.reshape(states.shape[0], num_repeat, 1)
        next_pi_values2 = next_pi_values2.reshape(states.shape[0], num_repeat, 1)
        
        cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
        cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)
        
        assert cat_q1.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q1 instead has shape: {cat_q1.shape}"
        assert cat_q2.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q2 instead has shape: {cat_q2.shape}"
        
        cql1_scaled_loss = ((torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q1_values.mean()) * self.cql_weight
        cql2_scaled_loss = ((torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q2_values.mean()) * self.cql_weight 
        
        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])
        # argmax vim baseline
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)
            
            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (-cql1_scaled_loss - cql2_scaled_loss) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()
        
        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss
        
        total_loss = total_c1_loss + total_c2_loss
            
        # Update critics
        self.critic_optim.zero_grad()
        total_loss.backward()
        self.critic_optim.step()

        soft_update(self.critic1_net_target, self.critic1_net, self.tau)
        soft_update(self.critic2_net_target, self.critic2_net, self.tau)
        
        avg_q_values = torch.mean(torch.min(q1_values, q2_values))
        return total_c1_loss.item(), total_c2_loss.item(), actor_loss.item(), avg_q_values.item()
    
    def save_models(self, epoch):
        self.actor_net.save_checkpoint(epoch)
        self.critic1_net.save_checkpoint(epoch)
        self.critic2_net.save_checkpoint(epoch)
        self.critic1_net_target.save_checkpoint(epoch)
        self.critic2_net_target.save_checkpoint(epoch)
        
    def load_models(self, epoch):
        self.actor_net.load_checkpoint(epoch)
        self.critic1_net.load_checkpoint(epoch)
        self.critic2_net.load_checkpoint(epoch)
        self.critic1_net_target.load_checkpoint(epoch)
    
    