import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import os

# Add project root to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tradenet.config import AGENT_DEFAULTS, EXPLORATION_DEFAULTS
from tradenet.q_network import DQNNetwork
from tradenet.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,                    
        hidden_sizes=(128, 128),
        gamma=None,
        lr=None,
        batch_size=None,
        buffer_capacity=None,
        min_buffer_size=None,
        eps_start=None,
        eps_end=None,
        eps_decay_steps=None,
        target_update_freq=None,       
        device='cpu',
        config_override=None
    ):

        # Load defaults from config
        agent_config = AGENT_DEFAULTS.copy()
        exploration_config = EXPLORATION_DEFAULTS.copy()
        
        # Apply override if provided
        if config_override:
            agent_config.update(config_override)
        
        # Set parameters
        self.state_dim = state_dim
        self.action_dim = action_dim                    
        
        # Agent hyperparameters
        self.gamma = gamma if gamma is not None else agent_config['gamma']
        self.batch_size = batch_size if batch_size is not None else agent_config['batch_size']
        self.min_buffer_size = min_buffer_size if min_buffer_size is not None else agent_config['min_buffer_size']
        self.device = device
        
        # Exploration parameters
        self.epsilon = eps_start if eps_start is not None else exploration_config['eps_start']
        self.eps_start = self.epsilon
        self.eps_end = eps_end if eps_end is not None else exploration_config['eps_end']
        self.eps_decay_steps = eps_decay_steps if eps_decay_steps is not None else exploration_config['eps_decay_steps']
        
        # Network parameters
        hidden_sizes = hidden_sizes if hidden_sizes else agent_config['hidden_sizes']
        learning_rate = lr if lr is not None else agent_config['lr']
        buffer_capacity = buffer_capacity if buffer_capacity is not None else agent_config['buffer_capacity']
        
        # Q-Network
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_sizes).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Training statistics
        self.train_steps = 0
        self.total_steps = 0
        self.is_training = True                       
        
        # Target update freq (for DDQN compatibility, not used in DQN)
        self.target_update_freq = target_update_freq
        
        print(f"DQN Agent initialized:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Buffer capacity: {buffer_capacity}")
        print(f"  Min buffer size: {self.min_buffer_size}")
        print(f"  Hidden sizes: {hidden_sizes}")
        print(f"  Epsilon: {self.eps_start} → {self.eps_end} over {self.eps_decay_steps} steps")
        print(f"  Device: {device}")
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
    
    def select_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            
            # Exploration: random action
            action_index = random.randint(0, self.action_dim - 1)
        
        else:
            
            # Exploitation: best action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_index = q_values.argmax().item()
        
        return action_index
    
    def train_step(self):
        if len(self.replay_buffer) < self.min_buffer_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values: Q(s, a)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.train_steps += 1
        
        # Update epsilon with linear decay
        self._update_epsilon()
        
        return loss.item()                             
    
    def _update_epsilon(self):
        if self.total_steps < self.eps_decay_steps:
            
            # Linear decay
            decay_progress = self.total_steps / self.eps_decay_steps
            self.epsilon = self.eps_start - (self.eps_start - self.eps_end) * decay_progress
        else:
            self.epsilon = self.eps_end
    
    def set_train_mode(self):
        self.is_training = True
        self.q_network.train()
    
    def set_eval_mode(self):
        self.is_training = False
        self.q_network.eval()
    
    def save(self, filepath):

        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
            'total_steps': self.total_steps,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,          
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'eps_start': self.eps_start,
                'eps_end': self.eps_end,
                'eps_decay_steps': self.eps_decay_steps,
            }
        }, filepath)
    
    def load(self, filepath, load_optimizer=True):
 
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        
        if 'train_steps' in checkpoint:
            self.train_steps = checkpoint['train_steps']
        
        if 'total_steps' in checkpoint:
            self.total_steps = checkpoint['total_steps']
        
        print(f"Loaded checkpoint from {filepath}")
        print(f"  Epsilon: {self.epsilon:.4f}")
        print(f"  Training steps: {self.train_steps}")
        print(f"  Total steps: {self.total_steps}")
    
    # Get Q-values for all actions for a given state
    def get_action_distribution(self, state):

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        return q_values
    
    def get_config(self):
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'min_buffer_size': self.min_buffer_size,
            'epsilon': self.epsilon,
            'eps_start': self.eps_start,
            'eps_end': self.eps_end,
            'eps_decay_steps': self.eps_decay_steps,
            'buffer_capacity': self.replay_buffer.buffer.maxlen,
            'train_steps': self.train_steps,
            'total_steps': self.total_steps,
        }