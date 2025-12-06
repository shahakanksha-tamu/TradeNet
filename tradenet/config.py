# config.py

import os

# Directory Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# All run outputs (logs, metrics, checkpoints)
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# Ensure the base directories exist
for d in [DATA_DIR, RAW_DATA_DIR, RUNS_DIR]:
    os.makedirs(d, exist_ok=True)

# Default Environment Hyperparameters
ENV_DEFAULTS = {
    "initial_amount": 100_000.0,      # initial capital
    "hmax": 1,                        # default action granularity - one share per trade
    "buy_cost_pct": 0.001,            # 0.1% transaction fee
    "sell_cost_pct": 0.001,           # 0.1% transaction fee
    "reward_scaling": 1e-4,
    "tech_indicator_list": ["ma20", "ma50", "vol20"],
    "window_size": 20,
    "allow_short": False,
    "max_pos": 1000,           
}

# Default Agent Hyperparameters (DDQN/DQN)
AGENT_DEFAULTS = {
    "gamma": 0.99,
    "lr": 1e-3,
    "batch_size": 64,
    "buffer_capacity": 100_000,
    "min_buffer_size": 1_000,
    "tau": 0.005,                     
    "hidden_sizes": (128, 128),      
}

# Default Exploration (ε-greedy)
EXPLORATION_DEFAULTS = {
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 50_000,
}


# Training Loop Defaults
TRAINING_DEFAULTS = {
    "num_episodes": 250,
    "seed": 0,
    "eval_interval": 5,
}

