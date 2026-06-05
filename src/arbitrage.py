import numpy as np
import pandas as pd
from .black_scholes import bs_price


def check_calendar(iv_surface):
    violations = []
    for strike, group in iv_surface.groupby("strike"):
        group = group.sort_values("T").reset_index(drop=True)
        total_var = group["IV"] ** 2 * group["T"]
        for i in range(1, len(total_var)):
            if total_var.iloc[i] < total_var.iloc[i - 1]:
                violations.append({"strike": strike, "T": group["T"].iloc[i], "type": "calendar"})
    return violations


def check_butterfly(iv_surface, T, S, r):
    slice_df = iv_surface[iv_surface["T"] == T].sort_values("strike").reset_index(drop=True)
    if len(slice_df) < 3:
        return []

    strikes = slice_df["strike"].values
    prices  = np.array([bs_price(S, K, T, r, iv) for K, iv in zip(strikes, slice_df["IV"].values)])

    violations = []
    for i in range(1, len(strikes) - 1):
        K1, K2, K3 = strikes[i - 1], strikes[i], strikes[i + 1]
        C1, C2, C3 = prices[i - 1], prices[i], prices[i + 1]
        # General convexity for non-uniform spacing: weighted butterfly spread
        w = (K3 - K2) / (K3 - K1)
        butterfly = w * C1 + (1 - w) * C3 - C2
        if butterfly < -0.02 * C2:
            violations.append({"strike": K2, "T": T, "type": "butterfly"})
    return violations
