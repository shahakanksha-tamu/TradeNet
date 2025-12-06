import argparse
import csv
import os
import random
import time
import numpy as np
import torch

from tradenet.logger import create_logger
from tradenet.env_single import SingleStockTradingEnv
from tradenet.agent_ddqn import DDQNAgent
from tradenet.agent_dqn import DQNAgent

from tradenet.config import (
    ENV_DEFAULTS,
    AGENT_DEFAULTS,
    EXPLORATION_DEFAULTS,
    TRAINING_DEFAULTS,
    RUNS_DIR,
)

from tradenet.utils import (
    load_raw_ticker_data,
    preprocess_raw_data,
    add_return_trend,
    train_test_val_split,
)


def set_global_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

# Run one full episode in evaluation mode (greedy policy, no learning)
def run_episode(env, agent):
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.select_action(state, explore=False)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state

    portfolio_values = env.get_portfolio_values()
    return total_reward, portfolio_values

# Compute the annualized Sharpe ratio from a sequence of portfolio values.
# Assumes daily data (252 trading days per year).
def compute_sharpe(portfolio_values):
    
    pv = np.array(portfolio_values, dtype=np.float64)
    if len(pv) < 2:
        return 0.0

    returns = pv[1:] / pv[:-1] - 1.0
    mean_ret = returns.mean()
    std_ret = returns.std()

    if std_ret <= 0:
        return 0.0

    sharpe = (mean_ret / std_ret) * np.sqrt(252)
    return float(sharpe)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN / DDQN agent on test data.")

    # Data / env config
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--initial_amount", type=float, default=ENV_DEFAULTS["initial_amount"])
    parser.add_argument("--hmax", type=int, default=ENV_DEFAULTS["hmax"])
    parser.add_argument("--buy_cost_pct", type=float, default=ENV_DEFAULTS["buy_cost_pct"])
    parser.add_argument("--sell_cost_pct", type=float, default=ENV_DEFAULTS["sell_cost_pct"])
    parser.add_argument("--reward_scaling", type=float, default=ENV_DEFAULTS["reward_scaling"])
    parser.add_argument("--window_size", type=int, default=ENV_DEFAULTS["window_size"])
    parser.add_argument("--max_pos", type=int, default=ENV_DEFAULTS["max_pos"])

    # Agent hyperparameters
    parser.add_argument("--gamma", type=float, default=AGENT_DEFAULTS["gamma"])
    parser.add_argument("--lr", type=float, default=AGENT_DEFAULTS["lr"])
    parser.add_argument("--batch_size", type=int, default=AGENT_DEFAULTS["batch_size"])
    parser.add_argument("--buffer_capacity", type=int, default=AGENT_DEFAULTS["buffer_capacity"])
    parser.add_argument("--min_buffer_size", type=int, default=AGENT_DEFAULTS["min_buffer_size"])
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=list(AGENT_DEFAULTS["hidden_sizes"]))

    # Exploration
    parser.add_argument("--eps_start", type=float, default=EXPLORATION_DEFAULTS["eps_start"])
    parser.add_argument("--eps_end", type=float, default=EXPLORATION_DEFAULTS["eps_end"])
    parser.add_argument("--eps_decay_steps", type=int, default=EXPLORATION_DEFAULTS["eps_decay_steps"])

    # Training control
    parser.add_argument("--num_episodes", type=int, default=TRAINING_DEFAULTS["num_episodes"])
    parser.add_argument("--algo", type=str, choices=["dqn", "ddqn"], default="ddqn")
    parser.add_argument("--seed", type=int, default=TRAINING_DEFAULTS["seed"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")


    args = parser.parse_args()


    # Setup: seed, run directories, logger

    set_global_seed(args.seed, args.device)

    run_name = (
        f"{args.algo}"
        f"_ticker-{args.ticker}"
        f"_hmax-{args.hmax}"
        f"_ep-{args.num_episodes}"
        f"_seed-{args.seed}"
    )

    run_dir = os.path.join(RUNS_DIR, run_name)
    logs_dir = os.path.join(run_dir, "logs")
    metrics_dir = os.path.join(run_dir, "metrics")
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    logger = create_logger(run_name, logs_dir, filename="eval.log")

    logger.info("========================================")
    logger.info(f"Evaluating run: {run_name}")
    logger.info(f"Device: {args.device}")
    logger.info("========================================")


    # Load data and build test environment

    df_raw = load_raw_ticker_data(args.ticker)
    df_feat = preprocess_raw_data(df_raw)
    df_feat = add_return_trend(df_feat, window=args.window_size)
    splits = train_test_val_split(df_feat)

    test_df = splits["test_df"]
    tech_indicator_list = ENV_DEFAULTS["tech_indicator_list"]

    test_env = SingleStockTradingEnv(
        df=test_df,
        initial_amount=args.initial_amount,
        hmax=args.hmax,
        buy_cost_pct=args.buy_cost_pct,
        sell_cost_pct=args.sell_cost_pct,
        reward_scaling=args.reward_scaling,
        tech_indicator_list=tech_indicator_list,
        window_size=args.window_size,
        max_pos=args.max_pos,
    )

    state_dim = test_env.reset().shape[0]
    action_dim = test_env.n_actions
    logger.info(f"Test env: state_dim={state_dim}, action_dim={action_dim}")

 
    # Build agent and load checkpoint

    hidden_sizes = tuple(args.hidden_sizes)
    AgentClass = DQNAgent if args.algo == "dqn" else DDQNAgent
   
    agent = AgentClass(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        min_buffer_size=args.min_buffer_size,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        target_update_freq=1000,
        device=args.device,
    )

    checkpoint_path = os.path.join(ckpt_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        logger.info(f"No checkpoint found at {checkpoint_path}. Please train the agent first.")
        return

    agent.load(checkpoint_path)
    agent.set_eval_mode()
    logger.info(f"Loaded checkpoint from {checkpoint_path}")


    # Run one evaluation episode on test set

    start_time = time.time()
    total_reward, pv = run_episode(test_env, agent)
    elapsed = time.time() - start_time

    final_value = float(pv[-1])
    initial_capital = args.initial_amount
    total_return = (final_value / initial_capital) - 1.0
    sharpe = compute_sharpe(pv)

    logger.info(f"Test episode finished in {elapsed:.2f} seconds.")
    logger.info(f"Final portfolio value: {final_value:.2f}")
    logger.info(f"Total return: {total_return:.4f}")
    logger.info(f"Sharpe ratio: {sharpe:.4f}")


    # Save equity curve and summary metrics
  
    # Equity curve
    test_equity_path = os.path.join(metrics_dir, "test_equity.csv")
    with open(test_equity_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "portfolio_value"])
        for i, v in enumerate(pv):
            writer.writerow([i, v])

    # Summary metrics
    test_summary_path = os.path.join(metrics_dir, "test_summary.csv")
    write_header = not os.path.exists(test_summary_path)
    with open(test_summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["algo", "ticker", "seed", "final_value", "total_return", "sharpe"])
        writer.writerow(
            [
                args.algo,
                args.ticker,
                args.seed,
                final_value,
                total_return,
                sharpe,
            ]
        )

    logger.info(f"Saved test equity curve to: {test_equity_path}")
    logger.info(f"Saved test summary metrics to: {test_summary_path}")


if __name__ == "__main__":
    main()
