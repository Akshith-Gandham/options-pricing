import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from pathlib import Path
from .black_scholes import implied_vol


def fetch_chain(ticker="^SPX", r=0.05, cache_path="data/options_chain.csv", force_refresh=False,
                min_T=7/365, max_T=2.0, n_expiries=8):
    cache = Path(cache_path)
    if cache.exists() and not force_refresh:
        return pd.read_csv(cache)

    stock  = yf.Ticker(ticker)
    S      = stock.history(period="1d")["Close"].iloc[-1]
    today  = pd.Timestamp.today().normalize()
    rows   = []

    # Collect all expiries in [min_T, max_T] then take n_expiries evenly spaced
    # across the time axis so we always capture short-, mid-, and long-dated options.
    candidates = []
    for expiry in stock.options:
        T = (pd.Timestamp(expiry) - today).days / 365
        if min_T <= T <= max_T:
            candidates.append((expiry, T))

    if not candidates:
        return pd.DataFrame()

    if len(candidates) <= n_expiries:
        selected = candidates
    else:
        step  = (len(candidates) - 1) / (n_expiries - 1)
        idx   = [round(i * step) for i in range(n_expiries)]
        selected = [candidates[i] for i in idx]

    for expiry, T in selected:
        calls = stock.option_chain(expiry).calls
        calls = calls[calls["bid"] > 0].copy()
        calls["mid"] = (calls["bid"] + calls["ask"]) / 2
        for _, row in calls.iterrows():
            iv = implied_vol(row["mid"], S, row["strike"], T, r)
            if np.isnan(iv) or not (0.01 < iv < 5.0):
                continue
            rows.append({
                "strike":      row["strike"],
                "T":           T,
                "IV":          iv,
                "mid":         row["mid"],
                "moneyness":   np.log(row["strike"] / S),
                "spot":        S,
            })

    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def plot_iv_surface(df):
    fig = go.Figure(data=[go.Scatter3d(
        x=df["moneyness"],
        y=df["T"],
        z=df["IV"],
        mode="markers",
        marker=dict(size=3, color=df["IV"], colorscale="Viridis", showscale=True),
    )])
    fig.update_layout(
        title="SPX Implied Volatility Surface",
        scene=dict(
            xaxis_title="Log-Moneyness  ln(K/S)",
            yaxis_title="Maturity (years)",
            zaxis_title="Implied Vol",
        ),
    )
    fig.show()
