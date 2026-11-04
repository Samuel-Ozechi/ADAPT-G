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
        if any(window <= 0 for window in cfg.windows_seconds):
            raise ValueError("Rolling windows must be positive seconds")
        if cfg.min_history < 0:
            raise ValueError("min_history must be non-negative")
        self.cfg = cfg

    def fit_transform(self, df: pd.DataFrame, id_col: str = "TransactionID") -> pd.DataFrame:
        cfg = self.cfg
        out = df.copy()
        # Sort deterministically; preserve original index for safe re-join.
        out = out.reset_index().rename(columns={"index": "__orig_idx"})
        out = out.sort_values([cfg.entity_col, cfg.time_col, id_col],
                              kind="mergesort").reset_index(drop=True)

        # All aggregates exclude every transaction at the current timestamp,
        # not merely the current row. This also avoids cross-entity leakage
        # from shifting a globally concatenated grouped cumulative sum.
        group_positions = out.groupby(cfg.entity_col, sort=False).indices.values()
        times = out[cfg.time_col].to_numpy()
        amounts = out[cfg.amount_col].astype(float).to_numpy()
        prior_count = np.zeros(len(out), dtype=np.int32)
        prior_sum = np.zeros(len(out), dtype=float)
        window_counts = {
            w: np.zeros(len(out), dtype=np.int32) for w in cfg.windows_seconds
        }
        window_sums = {
            w: np.zeros(len(out), dtype=float) for w in cfg.windows_seconds
        }

        for positions in group_positions:
            entity_times = times[positions]
            entity_amounts = np.nan_to_num(amounts[positions], nan=0.0)
            prefix = np.concatenate(([0.0], np.cumsum(entity_amounts)))
            # searchsorted(..., left) makes all equal-time rows share the
            # same history endpoint, strictly before their timestamp.
            right = np.searchsorted(entity_times, entity_times, side="left")
            prior_count[positions] = right
            prior_sum[positions] = prefix[right]
            for w in cfg.windows_seconds:
                left = np.searchsorted(entity_times, entity_times - w, side="right")
                window_counts[w][positions] = right - left
                window_sums[w][positions] = prefix[right] - prefix[left]

        out["prior_count"] = prior_count
        out["prior_amt_sum"] = prior_sum

        for w in cfg.windows_seconds:
            col_cnt = f"prior_count_{w}s"
            col_sum = f"prior_amt_sum_{w}s"
            out[col_cnt] = window_counts[w]
            out[col_sum] = window_sums[w]
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

