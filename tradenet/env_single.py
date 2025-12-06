import numpy as np
import pandas as pd
from typing import Optional, List

class SingleStockTradingEnv:
    def __init__(
        self,
        df: pd.DataFrame,
        initial_amount: float,
        hmax: int,
        buy_cost_pct: float,
        sell_cost_pct: float,
        reward_scaling: float,
        tech_indicator_list,
        window_size: int = 20,
        max_pos: Optional[int] = None,
    ):
      
        self.df = df.reset_index(drop=True)
        self.initial_amount = float(initial_amount)
        self.hmax = int(hmax)
        self.buy_cost_pct = float(buy_cost_pct)
        self.sell_cost_pct = float(sell_cost_pct)
        self.reward_scaling = float(reward_scaling)
        self.tech_indicator_list = list(tech_indicator_list)
        self.window_size = int(window_size)
        self.max_pos = int(max_pos) if max_pos is not None else None

        # Number of discrete actions: [-hmax, ..., 0, ..., +hmax]
        self.n_actions = 2 * self.hmax + 1

        # Internal episode state
        self.current_step = None
        self.cash = None
        self.shares_held = None
        self.total_asset_value = None
        self.init_price = None
        self.avg_entry_price = None

        # For evaluation and analysis
        self.asset_memory = []
        self.action_memory = []

        self._check_columns()

    def _check_columns(self):
        required_cols = ["Close", "log_return"]
        required_cols.extend(self.tech_indicator_list)
        required_cols.extend([f"trend_{i}" for i in range(self.window_size)])

        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise KeyError(
                f"DataFrame is missing required columns for the environment: {missing}. "
            )
    
    def reset(self):
        self.current_step = 0
        self.cash = self.initial_amount
        self.shares_held = 0
        self.init_price = float(self.df.loc[self.current_step, "Close"])
        self.total_asset_value = self.cash
        self.avg_entry_price = 0.0

        self.asset_memory = [self.total_asset_value]
        self.action_memory = []

        return self._get_observation()

    def _get_price(self, step: Optional[int] = None):
        if step is None:
            step = self.current_step
        return float(self.df.loc[step, "Close"])

    def _get_observation(self):
        row = self.df.loc[self.current_step]

        # Trend window features
        trend_feats = [float(row[f"trend_{i}"]) for i in range(self.window_size)]

        # Technical indicators
        ind_feats = [float(row[col]) for col in self.tech_indicator_list]

        # Scalar features
        current_price = float(row["Close"])
        price_rel = current_price / self.init_price if self.init_price and self.init_price > 0 else 1.0

        # Cash Norm
        cash_norm = (
            self.cash / self.initial_amount
            if self.cash is not None and self.initial_amount > 0
            else 0.0
        )

        # Position Norm
        if self.max_pos is not None and self.max_pos > 0:
            position_norm = (self.shares_held or 0) / self.max_pos
        else:
            # fallback normalization if max_pos is not used
            position_norm = 0.0 if not self.shares_held else float(self.shares_held)
        
        # Unrealized PnL
        if self.shares_held and self.shares_held > 0 and self.avg_entry_price > 0:
            unrealized_pnl_pct = (current_price - self.avg_entry_price) / self.avg_entry_price
        else:
            unrealized_pnl_pct = 0.0

        # Final state vector
        state = np.array(
            trend_feats + ind_feats + [price_rel, cash_norm, position_norm, unrealized_pnl_pct],
            dtype=np.float32,
        )

        return state
    
    def step(self, action):
      
        if not (0 <= action < self.n_actions):
            raise ValueError(f"Action {action} is out of bounds for n_actions={self.n_actions}.")

        # Map action index to change in number of shares
        delta_shares = action - self.hmax

        prev_total_value = self.total_asset_value
        current_price = self._get_price()

        # Buy action
        if delta_shares > 0:
            
            # Budget constraint
            max_affordable = int(self.cash // (current_price * (1 + self.buy_cost_pct)))
            shares_to_buy = min(delta_shares, max_affordable)

            # Maximum position constraint
            if self.max_pos is not None:
                allowed_by_position = self.max_pos - (self.shares_held or 0)
                shares_to_buy = min(shares_to_buy, max(0, allowed_by_position))

            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                fee = cost * self.buy_cost_pct
                self.cash -= (cost + fee)

                # Update average entry price using weighted average
                prev_shares = self.shares_held or 0
                total_shares = prev_shares + shares_to_buy

                if total_shares > 0:
                    total_cost_basis = prev_shares * self.avg_entry_price + shares_to_buy * current_price
                    self.avg_entry_price = total_cost_basis / total_shares
                else:
                    self.avg_entry_price = 0.0

                self.shares_held = total_shares

        # Sell action
        elif delta_shares < 0:

            # No short selling: can only sell what we hold
            prev_shares = self.shares_held or 0
            shares_to_sell = min(-delta_shares, prev_shares)

            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price
                fee = revenue * self.sell_cost_pct
                self.cash += (revenue - fee)
                self.shares_held = prev_shares - shares_to_sell

                # If position fully closed, reset avg_entry_price
                if self.shares_held == 0:
                    self.avg_entry_price = 0.0

        # Advance time
        self.current_step += 1
        done = False
        
        if self.current_step >= len(self.df) - 1:
            self.current_step = len(self.df) - 1
            done = True

        new_price = self._get_price()
        self.total_asset_value = self.cash + (self.shares_held or 0) * new_price

        # Reward: change in portfolio value scaled
        reward = (self.total_asset_value - prev_total_value) * self.reward_scaling

        # Book-keeping
        self.asset_memory.append(self.total_asset_value)
        self.action_memory.append(delta_shares)

        info = {
            "step": self.current_step,
            "price": new_price,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "total_asset_value": self.total_asset_value,
            "avg_entry_price": self.avg_entry_price,
        }

        next_state = self._get_observation()
        return next_state, float(reward), done, info

    # Return the recorded portfolio values over the episode
    def get_portfolio_values(self):
        return self.asset_memory
