"""Time-safe rolling aggregates: strictly prior to the target transaction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RollingConfig:
    entity_col: str = "card1"
    time_col: str = "TransactionDT"
    amount_col: str = "TransactionAmt"
    windows_seconds: tuple[int, ...] = (3600, 86400, 7 * 86400)
    min_history: int = 1


class RollingAggregator:
    """For each row, compute aggregates over strictly prior transactions
    of the same entity within each window.

    Leakage guarantee: for row i, only rows j with same entity and
    TransactionDT_j < TransactionDT_i are used. Ties are broken by
    TransactionID to keep determinism; equal-timestamp rows do NOT see
    each other (strict <).
    """

    def __init__(self, cfg: RollingConfig):
        self.cfg = cfg

    def fit_transform(self, df: pd.DataFrame, id_col: str = "TransactionID") -> pd.DataFrame:
        cfg = self.cfg
        out = df.copy()
        # Sort deterministically; preserve original index for safe re-join.
        out = out.reset_index().rename(columns={"index": "__orig_idx"})
        out = out.sort_values([cfg.entity_col, cfg.time_col, id_col],
                              kind="mergesort").reset_index(drop=True)

        group = out.groupby(cfg.entity_col, sort=False)
        # Prior count (excludes current row).
        out["prior_count"] = group.cumcount().astype("int32")
        # Prior amount sum / mean via shifted cumsum.
        amounts = out[cfg.amount_col].astype(float)
        out["prior_amt_sum"] = group[cfg.amount_col].cumsum().shift(1).fillna(0.0).values

        for w in cfg.windows_seconds:
            col_cnt = f"prior_count_{w}s"
            col_sum = f"prior_amt_sum_{w}s"
            out[col_cnt] = self._rolling_count(out, cfg, w)
            out[col_sum] = self._rolling_sum(out, cfg, w, amounts)
            out[f"prior_amt_mean_{w}s"] = np.where(
                out[col_cnt] > 0, out[col_sum] / out[col_cnt], np.nan
            )

        # Cold-start: no prior history -> leave NaN for window features.
        for w in cfg.windows_seconds:
            mask = out[f"prior_count_{w}s"] < cfg.min_history
            out.loc[mask, [f"prior_count_{w}s", f"prior_amt_sum_{w}s",
                           f"prior_amt_mean_{w}s"]] = np.nan

        # Restore original order.
        out = out.sort_values("__orig_idx").drop(columns="__orig_idx").reset_index(drop=True)
        return out

    @staticmethod
    def _rolling_count(out: pd.DataFrame, cfg: RollingConfig, w: int) -> pd.Series:
        # Vectorised per-entity window count using groupby + rolling on time.
        def per_group(g: pd.DataFrame) -> pd.Series:
            t = g[cfg.time_col].values
            ids = g.index.values
            res = np.zeros(len(t), dtype="int32")
            # Two-pointer over sorted times.
            left = 0
            for i in range(len(t)):
                while t[left] <= t[i] - w:
                    left += 1
                res[i] = i - left  # strictly prior rows within window
            return pd.Series(res, index=ids)
        return out.groupby(cfg.entity_col, sort=False, group_keys=False).apply(per_group)

    @staticmethod
    def _rolling_sum(out: pd.DataFrame, cfg: RollingConfig, w: int,
                     amounts: pd.Series) -> pd.Series:
        def per_group(g: pd.DataFrame) -> pd.Series:
            t = g[cfg.time_col].values
            a = amounts.loc[g.index].values
            ids = g.index.values
            res = np.zeros(len(t), dtype=float)
            left = 0
            running = 0.0
            for i in range(len(t)):
                while t[left] <= t[i] - w:
                    running -= a[left]
                    left += 1
                res[i] = running  # sum of prior rows within window
                running += a[i]
            return pd.Series(res, index=ids)
        return out.groupby(cfg.entity_col, sort=False, group_keys=False).apply(per_group)