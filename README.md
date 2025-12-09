# TradeNet - Deep Reinforcement Learning for Single-Stock Trading

This project implements a Deep Reinforcement Learning (DRL) trading system for a **single-stock environment**, using **Double Deep Q-Learning (DDQN)**. The agent learns to trade (buy, sell, hold) using windowed historical data, technical indicators, and a multi-share discrete action space. The goal is to **maximize risk-adjusted returns**, measured primarily through the **Sharpe ratio**. The agent is trained using a Double Deep Q-Network (DDQN) inside a custom trading environment.

The repository includes:
- A script to fetch raw stock data from Yahoo Finance
- A training pipeline
- A testing/evaluation pipeline
- A hyperparameter tuning script
- Visualiztion script

## Project Structure 

data/raw/                # Raw stock CSV data
scripts/fetch_data.py    # Script to fetch stock data
tradenet/                # Specifies environment, agents, utilities
train.py                 # Trains the DDQN agent
eval.py                  # Evaluates on test data
hypertune.py             # Runs Optuna hyperparameter tuning
requirements.txt         # Package Installation Requirements


## Setup Instructions

### 1. Install dependencies
pip install -r requirements.txt

## Fetch Stock Data (Required Before Training)

Use the provided script to download data:

```
python3 scripts/fetch_data.py --ticker AAPL --start YY-MM-DD --end YY-MM-DD
```

This saves the file:
data/raw/AAPL.csv

## Train the Agent

python3 train.py --ticker AAPL

You may also customize parameters for example:

```
python3 eval.py \
  --ticker AAPL \
  --hmax 100 \
  --lr 0.0001 \
  --gamma 0.95 \
  --num_episodes 150 \
  --seed 42
```

Training outputs (logs, checkpoints, metrics) will be stored in the runs/ folder.

## Evaluate the Trained Model

Be sure to pass the same parameters used while training to generate test results
```
python3 eval.py \
  --ticker AAPL \
  --hmax 100 \
  --lr 0.0001 \
  --gamma 0.95 \
  --num_episodes 150 \
  --seed 42
```

This returns:
- Final portfolio value
- Total return
- Sharpe ratio

And generates below graphs and metrics:
- test_equity.csv
- test_summary.csv

## Hyperparameter Tuning

```
python3 hypertune.py
```

This runs Optuna to search for the best hyperparameters and saves results to:
hyperstudy.pkl


## Visualization

```
python3 visualize.py --run_dir <relative path to run folder generated while training and evaluation>
```

This creates the visualization graphs such as the test-equity curve, train-validation curve, TD-Lose graph and comparison of agent performance against simple baseline buy-and-hold startegy.
