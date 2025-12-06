import torch.nn as nn

class DQNNetwork(nn.Module):
    """Deep Q-Network for stock trading."""
    
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            hidden_sizes: Tuple of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state)