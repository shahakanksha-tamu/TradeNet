import argparse
import csv
import os
import random
import numpy as np
import torch
import time
from tradenet.logger import create_logger
from tradenet.env_single import SingleStockTradingEnv
from tradenet.agent_ddqn import DDQNAgent
from tradenet.agent_dqn import DQNAgent

from tradenet.config import (
    ENV_DEFAULTS,
    AGENT_DEFAULTS,
    EXPLORATION_DEFAULTS,
    TRAINING_DEFAULTS,
    RUNS_DIR
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

   

# Run one full episode
# If train=True: epsilon-greedy and Q-updates.
# If train=False: greedy only, no learning.

def run_episode(env, agent, train = True):
    state = env.reset()
    done = False
    total_reward = 0.0
    losses = []

    while not done:
        action = agent.select_action(state, explore=train)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if train:
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(float(loss))

        state = next_state

    portfolio_values = env.get_portfolio_values()
    return total_reward, losses, portfolio_values

def main():

    parser = argparse.ArgumentParser()

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
    parser.add_argument("--eval_interval", type=int, default=TRAINING_DEFAULTS["eval_interval"])
    parser.add_argument("--algo", type=str, choices=["dqn", "ddqn"], default="ddqn")
    parser.add_argument("--seed", type=int, default=TRAINING_DEFAULTS["seed"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()


    # Setup: seeds, logger, dirs
    set_global_seed(args.seed, args.device)

    run_name = (
        f"{args.algo}"
        f"_ticker-{args.ticker}"
        f"_hmax-{args.hmax}"
        f"_ep-{args.num_episodes}"
        f"_seed-{args.seed}"
    )

    # Per-run directory layout
    run_dir = os.path.join(RUNS_DIR, run_name)
    logs_dir = os.path.join(run_dir, "logs")
    metrics_dir = os.path.join(run_dir, "metrics")
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    for d in [run_dir, logs_dir, metrics_dir, ckpt_dir]:
        os.makedirs(d, exist_ok=True)

    logger = create_logger(run_name, logs_dir, filename="train.log")

    logger.info("========================================")
    logger.info(f"Run: {run_name}")
    logger.info(f"Device: {args.device}")
    logger.info(
        f"Episodes={args.num_episodes}, eval_interval={args.eval_interval}, "
        f"gamma={args.gamma}, lr={args.lr}, batch_size={args.batch_size}"
    )
    logger.info(
        f"Env: initial_amount={args.initial_amount}, hmax={args.hmax}, "
        f"buy_cost_pct={args.buy_cost_pct}, sell_cost_pct={args.sell_cost_pct}, "
        f"window_size={args.window_size}, max_pos={args.max_pos}"
    )
    logger.info("========================================")


    # Load data, preprocess, and split
    df_raw = load_raw_ticker_data(args.ticker)
    df_feat = preprocess_raw_data(df_raw)
    df_feat = add_return_trend(df_feat, window=args.window_size)
    splits = train_test_val_split(df_feat)

    train_df = splits["train_df"]
    val_df = splits["val_df"]

    tech_indicator_list = ENV_DEFAULTS["tech_indicator_list"]

    # Build environement
    def make_env(df):
        
        return SingleStockTradingEnv(
            df=df,
            initial_amount=args.initial_amount,
            hmax=args.hmax,
            buy_cost_pct=args.buy_cost_pct,
            sell_cost_pct=args.sell_cost_pct,
            reward_scaling=args.reward_scaling,
            tech_indicator_list=tech_indicator_list,
            window_size=args.window_size,
            max_pos=args.max_pos,
        )
    
    # Train and validation environments
    train_env = make_env(train_df)
    val_env = make_env(val_df)

    state_dim = train_env.reset().shape[0]
    action_dim = train_env.n_actions
    logger.info(f"State dimension: {state_dim}, Action dimension: {action_dim}")


    # Build agent (DQN or DDQN)
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

    logger.info(
        f"Agent: {AgentClass.__name__} with hidden_sizes={hidden_sizes}, "
        f"buffer_capacity={args.buffer_capacity}, min_buffer_size={args.min_buffer_size}"
    )

  
    # Prepare CSV for episode metrics and checkpoint path
    train_csv_path = os.path.join(metrics_dir, "train_metrics.csv")
    with open(train_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "train_final_value",
            "train_total_return",
            "avg_td_loss",
            "val_final_value",
            "val_total_return",
        ])

    checkpoint_path = os.path.join(ckpt_dir, "best_model.pth")
    best_val_final = -np.inf
    start_time = time.time()
    initial_capital = args.initial_amount
    
    logger.info("Starting training.")

    # Training loop
    for ep in range(1, args.num_episodes + 1):

        # Training episode
        agent.set_train_mode()

        # accumulated reward, losses, and array of portfolio values
        train_reward, train_losses, train_pv = run_episode(train_env, agent, train=True)

        # final portfolio value at last timestep
        train_final_value = float(train_pv[-1])

        # Total return
        train_total_return = (train_final_value / initial_capital) - 1.0
        
        # average TD loss over training episode
        avg_td_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # Evaluation on validation environment
        val_final_value = np.nan
        val_total_return = np.nan

        if ep % args.eval_interval == 0:
            agent.set_eval_mode()
            val_reward, _, val_pv = run_episode(val_env, agent, train=False)
            val_final_value = float(val_pv[-1])
            val_total_return = (val_final_value / initial_capital) - 1.0

            # Track best validation performance
            if val_final_value > best_val_final:
                best_val_final = val_final_value
                agent.save(checkpoint_path)
                saved_str = " (saved best)"
            else:
                saved_str = ""
        else:
            saved_str = ""

        # Log per-episode metrics to CSV
        with open(train_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep,
                train_final_value,
                train_total_return,
                avg_td_loss,
                val_final_value,
                val_total_return,
            ])

        # Progress log
        log_msg = (
            f"Ep {ep:03d}/{args.num_episodes} | "
            f"train_final={train_final_value:.2f} "
            f"(ret={train_total_return:.4f}) | "
            f"avg_td_loss={avg_td_loss:.4f}"
        )
        if not np.isnan(val_final_value):
            log_msg += (
                f" | val_final={val_final_value:.2f} "
                f"(ret={val_total_return:.4f}){saved_str}"
            )

        logger.info(log_msg)

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed / 60:.2f} minutes.")
    logger.info(f"Best validation final value: {best_val_final:.2f}")
    logger.info(f"Best model checkpoint: {checkpoint_path}")
    logger.info(f"Training metrics CSV: {train_csv_path}")


if __name__ == "__main__":
    main()