import numpy as np


def mc_price(S, K, T, r, sigma, option_type="european_call", n_paths=100_000, n_steps=252):
    dt = T / n_steps
    Z = np.random.standard_normal((n_paths, n_steps))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths = S * np.exp(np.cumsum(log_returns, axis=1))
    paths = np.hstack([np.full((n_paths, 1), S), paths])

    if option_type == "european_call":
        payoffs = np.maximum(paths[:, -1] - K, 0)
    elif option_type == "asian_call":
        payoffs = np.maximum(paths.mean(axis=1) - K, 0)
    elif option_type == "barrier_down_out_call":
        barrier = K * 0.85
        alive  = paths.min(axis=1) >= barrier
        payoffs = np.maximum(paths[:, -1] - K, 0) * alive
    else:
        raise ValueError(f"Unknown option_type: {option_type}")

    price   = np.exp(-r * T) * payoffs.mean()
    std_err = payoffs.std() / np.sqrt(n_paths)
    return price, std_err
