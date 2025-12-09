import torch.nn as nn

class DQNNetwork(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):

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
    
    #Forward pass through the network
    def forward(self, state):
   
        return self.network(state)