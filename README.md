# Options Pricing & Volatility Surface Calibration

End-to-end implementation of options pricing, implied volatility surface construction, and stochastic volatility calibration.

## What's covered

| Step | Topic |
|------|-------|
| 1 | Black-Scholes pricer + full Greeks + implied volatility (Brent's method) |
| 2 | Monte Carlo pricer — European, Asian, Barrier (down-and-out) |
| 3 | SPX implied volatility surface from live market data (yfinance) |
| 4 | Heston stochastic volatility model — CF pricing + calibration |
| 5 | Arbitrage detection — calendar spread and butterfly violations |

## Heston implementation

Uses the **Little Heston Trap** formulation (Albrecher et al. 2007) which avoids the branch-cut discontinuities in the complex logarithm that appear in the original 1993 characteristic function. Pricing uses Gil-Pelaez Fourier inversion with a vectorized numpy quadrature grid — all strikes at a given maturity are priced in one pass, making calibration fast. Calibration minimizes price MSE (not IV MSE) to avoid double-inversion noise.

## Arbitrage detection

Butterfly violations are flagged when the breach exceeds 2% of the option price — approximately 2× the typical SPX bid-ask spread. Smaller breaches are mid-price noise from discrete strikes, not tradeable arbitrage.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook demo.ipynb
```

Option chain data is fetched from yfinance on the first run and cached to `data/options_chain.csv`. The fetch selects 8 expirations evenly spaced across the full available maturity range (1 week to 2 years), so the surface always spans short-, mid-, and long-dated options. To re-pull live data, pass `force_refresh=True` to `fetch_chain`.

## Project layout

```
src/
  black_scholes.py   # bs_price, greeks, implied_vol
  monte_carlo.py     # mc_price (European / Asian / Barrier)
  iv_surface.py      # fetch_chain, plot_iv_surface
  heston.py          # heston_cf, heston_price, calibrate_heston
  arbitrage.py       # check_calendar, check_butterfly
data/                # cached option chains (git-ignored)
demo.ipynb           # narrative walkthrough with plots
```
