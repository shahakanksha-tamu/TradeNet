import random
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from optuna.importance import get_param_importances
from tqdm.auto import tqdm 

from tradenet.config import ENV_DEFAULTS, AGENT_DEFAULTS, EXPLORATION_DEFAULTS
from tradenet.env_single import SingleStockTradingEnv
from tradenet.agents.agent_ddqn import DDQNAgent
from tradenet.utils import (
    load_raw_ticker_data,
    preprocess_raw_data,
    add_return_trend,
    train_test_val_split,
)

def set_global_seed(seed: int, device: str):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def run_episode(env, agent, train: bool = True):

    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state, explore=train)
        next_state, reward, done, info = env.step(action)

        if train:
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()

        state = next_state

    pv = env.get_portfolio_values()
    return pv


def compute_sharpe(portfolio_values, trading_days_per_year = 252):
    pv = np.array(portfolio_values, dtype=np.float64)
    if len(pv) < 2:
        return 0.0

    returns = pv[1:] / pv[:-1] - 1.0
    mean_ret = returns.mean()
    std_ret = returns.std()

    if std_ret <= 0:
        return 0.0

    sharpe = (mean_ret / std_ret) * np.sqrt(trading_days_per_year)
    return float(sharpe)


def objective(trial: optuna.Trial) -> float:

    # Hyperparameter space
    hmax = trial.suggest_categorical("hmax", [50, 80, 100, 120, 150])
    window_size = trial.suggest_categorical("window_size", [20, 50, 80, 100])
    lr = trial.suggest_categorical("lr", [1e-5, 5e-5, 1e-4, 1e-3])
    gamma = trial.suggest_categorical("gamma", [0.80, 0.85, 0.90, 0.95, 0.99])
    num_episodes = trial.suggest_categorical("num_episodes", [10, 50, 100, 150, 200])

    # Validation interval (episodes)
    K = 10  

    # Random seed for the trial
    seed = trial.suggest_int("seed", 0, 10_000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(seed, device)


    print("\n===============================================")
    print(f"### TRIAL {trial.number} STARTED ###")
    print("Chosen Hyperparameters:")
    print(f"  hmax          = {hmax}")
    print(f"  window_size   = {window_size}")
    print(f"  lr            = {lr}")
    print(f"  gamma         = {gamma}")
    print(f"  num_episodes  = {num_episodes}")
    print(f"  seed          = {seed}")
    print("===============================================\n")


    # Load & process data
    ticker = "AAPL"
    df_raw = load_raw_ticker_data(ticker)
    df_feat = preprocess_raw_data(df_raw)
    df_feat = add_return_trend(df_feat, window=window_size)
    splits = train_test_val_split(df_feat)

    train_df, val_df = splits["train_df"], splits["val_df"]

    tech_indicator_list = ENV_DEFAULTS["tech_indicator_list"]
    initial_capital = ENV_DEFAULTS["initial_amount"]

    # Build environments
    def make_env(df):
        return SingleStockTradingEnv(
            df=df,
            initial_amount=initial_capital,
            hmax=hmax,
            buy_cost_pct=ENV_DEFAULTS["buy_cost_pct"],
            sell_cost_pct=ENV_DEFAULTS["sell_cost_pct"],
            reward_scaling=ENV_DEFAULTS["reward_scaling"],
            tech_indicator_list=tech_indicator_list,
            window_size=window_size,
            max_pos=ENV_DEFAULTS["max_pos"],
        )

    train_env = make_env(train_df)
    val_env = make_env(val_df)

    state_dim = train_env.reset().shape[0]
    action_dim = train_env.n_actions


    # Build agent
    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=tuple(AGENT_DEFAULTS["hidden_sizes"]),
        gamma=gamma,
        lr=lr,
        batch_size=AGENT_DEFAULTS["batch_size"],
        buffer_capacity=AGENT_DEFAULTS["buffer_capacity"],
        min_buffer_size=AGENT_DEFAULTS["min_buffer_size"],
        eps_start=EXPLORATION_DEFAULTS["eps_start"],
        eps_end=EXPLORATION_DEFAULTS["eps_end"],
        eps_decay_steps=EXPLORATION_DEFAULTS["eps_decay_steps"],
        target_update_freq=1000,
        device=device,
    )


    # Training 
    best_val_sharpe = -999.0

    pbar = tqdm(
        range(1, num_episodes + 1),
        desc=f"Trial {trial.number} episodes",
        leave=False,
    )

    for ep in pbar:

        run_episode(train_env, agent, train=True)

        pbar.set_postfix({"ep": ep})
      
        if ep % K == 0 or ep == num_episodes:
            pv = run_episode(val_env, agent, train=False)
            val_sharpe = compute_sharpe(pv)


            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe

      
            pbar.set_postfix({
                "ep": ep,
                "val_sharpe": f"{val_sharpe:.3f}",
                "best": f"{best_val_sharpe:.3f}",
            })


            trial.report(best_val_sharpe, step=ep)

            if trial.should_prune():
                pbar.close()
                print(
                    f"[Trial {trial.number}] PRUNED at episode {ep} "
                    f"(best Sharpe = {best_val_sharpe:.4f})"
                )
                raise optuna.TrialPruned()

    pbar.close()
    print(
        f"[Trial {trial.number}] COMPLETED with Best Validation Sharpe = {best_val_sharpe:.4f}"
    )
    return best_val_sharpe



if __name__ == "__main__":

    random.seed(0)
    np.random.seed(0)

    pruner = optuna.pruners.MedianPruner(
        n_warmup_steps=5  
    )

    study = optuna.create_study(
        study_name="tradenet_ddqn_hypersearch",
        direction="maximize",
        pruner=pruner,
    )

    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print("\nBest trial:")
    best = study.best_trial
    print("  Sharpe:", best.value)
    print("  Params:", best.params)

    # Save the study
    joblib.dump(study, "hyperstudy.pkl")
    print("Saved Optuna study to hyperstudy.pkl")