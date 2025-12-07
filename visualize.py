import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from tradenet.utils import (
    load_raw_ticker_data,
    preprocess_raw_data,
    add_return_trend,
    train_test_val_split,
)

from tradenet.config import ENV_DEFAULTS

def load_run_config(run_dir):
    
    config_path = os.path.join(run_dir, "config.json")
    
    if not os.path.exists(config_path):
        print(f"[WARN] No config.json found in {run_dir}. Using defaults.")
        return None
    
    with open(config_path, "r") as f:
        return json.load(f)

# Parse run directory name
def parse_run_name(run_dir: str):
    
    base = os.path.basename(os.path.normpath(run_dir))
    parts = base.split("_")
    info = {"algo": None, "ticker": None, "hmax": None, "ep": None, "seed": None}

    if parts:
        info["algo"] = parts[0]

    for p in parts[1:]:
        if p.startswith("ticker-"):
            info["ticker"] = p.replace("ticker-", "")
        elif p.startswith("hmax-"):
            info["hmax"] = p.replace("hmax-", "")
        elif p.startswith("ep-"):
            info["ep"] = p.replace("ep-", "")
        elif p.startswith("seed-"):
            info["seed"] = p.replace("seed-", "")

    return info


def ensure_plots_dir(run_dir):
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

# Plot training and validation total returns vs episode
def plot_train_val_total_returns(train_csv_path, out_path):
 
    df = pd.read_csv(train_csv_path)

    episodes = df["episode"].values
    train_ret = df["train_total_return"].values
    val_ret = df["val_total_return"].values

    plt.figure()
    plt.plot(episodes, train_ret, label="Train total return")
    mask = ~np.isnan(val_ret)
    if mask.any():
        plt.plot(episodes[mask], val_ret[mask], label="Validation total return")
    plt.xlabel("Episode")
    plt.ylabel("Total return")
    plt.title("Training and validation total returns per episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# TD-Loss plot
def plot_td_loss(train_csv_path, out_path):
    
    df = pd.read_csv(train_csv_path)

    episodes = df["episode"].values
    avg_td_loss = df["avg_td_loss"].values

    # Skip plot if all losses are NaN
    if np.all(np.isnan(avg_td_loss)):
        return

    plt.figure()
    plt.plot(episodes, avg_td_loss)
    plt.xlabel("Episode")
    plt.ylabel("Average TD loss")
    plt.title("Average TD loss per episode")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# Plot the test equity curve (portfolio value vs step)
def plot_test_equity(equity_csv_path, out_path):
   
    df = pd.read_csv(equity_csv_path)

    steps = df["step"].values
    pv = df["portfolio_value"].values

    plt.figure()
    plt.plot(steps, pv)
    plt.xlabel("Step")
    plt.ylabel("Portfolio value")
    plt.title("Test equity curve (agent)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# Plot agent equity curve vs Buy-and-Hold baseline on the test period.
def plot_test_equity_vs_buyhold(equity_csv_path, ticker, initial_amount, out_path):

    # Agent equity from CSV
    df_eq = pd.read_csv(equity_csv_path)
    agent_pv = df_eq["portfolio_value"].values

    # Load full price series and recompute the same test split
    df_raw = load_raw_ticker_data(ticker)
    df_feat = preprocess_raw_data(df_raw)
    df_feat = add_return_trend(df_feat, window=ENV_DEFAULTS["window_size"])
    splits = train_test_val_split(df_feat)
    test_df = splits["test_df"]

    test_close = test_df["Close"].values.astype(float)
    if len(test_close) == 0:
        return

    # Buy-and-hold: invest all capital at first test day and hold
    bh_pv = initial_amount * (test_close / test_close[0])

    # Align lengths with agent curve, just in case of off-by-one
    L = min(len(agent_pv), len(bh_pv))
    steps = np.arange(L)
    agent_pv = agent_pv[:L]
    bh_pv = bh_pv[:L]

    plt.figure()
    plt.plot(steps, agent_pv, label="Agent")
    plt.plot(steps, bh_pv, label="Buy & Hold")
    plt.xlabel("Step")
    plt.ylabel("Portfolio value")
    plt.title(f"Test equity: Agent vs Buy & Hold ({ticker})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# visualize all plots for single run
def visualize_single_run(run_dir):
 
    metrics_dir = os.path.join(run_dir, "metrics")
    plots_dir = ensure_plots_dir(run_dir)

    train_csv_path = os.path.join(metrics_dir, "train_metrics.csv")
    
    if not os.path.exists(train_csv_path):
        print(f"[WARN] train_metrics.csv not found at {train_csv_path}, skipping training plots.")
        return

    metrics_dir = os.path.join(run_dir, "metrics")
    plots_dir = ensure_plots_dir(run_dir)

    initial_amount = None

    config = load_run_config(run_dir)
    initial_amount = config.get("initial_amount",  ENV_DEFAULTS["initial_amount"])
    
    # Parse run name to infer ticker; fallback to ENV_DEFAULTS if needed
    info = parse_run_name(run_dir)
    ticker = info["ticker"] if info["ticker"] is not None else "AAPL"

    # Training vs validation returns
    out_train_val = os.path.join(plots_dir, "train_val_returns.png")
    plot_train_val_total_returns(train_csv_path, out_train_val)
    print(f"[INFO] Saved: {out_train_val}")

    # TD loss
    out_loss = os.path.join(plots_dir, "td_loss.png")
    plot_td_loss(train_csv_path, out_loss)
    print(f"[INFO] Saved: {out_loss} (if not empty)")

    # Test equity curve and Agent vs Buy&Hold plots
    test_equity_path = os.path.join(metrics_dir, "test_equity.csv")
    if os.path.exists(test_equity_path):
        out_equity = os.path.join(plots_dir, "test_equity_curve.png")
        plot_test_equity(test_equity_path, out_equity)
        print(f"[INFO] Saved: {out_equity}")

        out_equity_bh = os.path.join(plots_dir, "test_equity_vs_buyhold.png")
        plot_test_equity_vs_buyhold(
            test_equity_path,
            ticker=ticker,
            initial_amount=initial_amount,
            out_path=out_equity_bh,
        )
        print(f"[INFO] Saved: {out_equity_bh}")
    else:
        print(f"[WARN] No test_equity.csv at {test_equity_path}; skipping test equity plots.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize all plots for a SINGLE run under runs/."
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to a run directory under runs/ (e.g. runs/ddqn_ticker-AAPL_hmax-1_ep-250_seed-0)",
    )

    args = parser.parse_args()
    visualize_single_run(args.run_dir)


if __name__ == "__main__":
    main()