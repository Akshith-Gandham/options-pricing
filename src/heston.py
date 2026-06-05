import numpy as np
from scipy.optimize import minimize

_QUAD_U = np.linspace(1e-5, 50, 200)   # fixed integration grid, computed once at import


def heston_cf(u, S, T, r, v0, kappa, theta, sigma_v, rho):
    # Little Heston Trap formulation (Albrecher et al. 2007) — avoids branch-cut
    # discontinuities in the complex log that appear in the original 1993 CF.
    x      = np.log(S) + r * T
    d      = np.sqrt((rho * sigma_v * 1j * u - kappa)**2 + sigma_v**2 * (1j * u + u**2))
    g      = (kappa - rho * sigma_v * 1j * u + d) / (kappa - rho * sigma_v * 1j * u - d)
    exp_dT = np.exp(d * T)
    C = (kappa * theta / sigma_v**2) * (
        (kappa - rho * sigma_v * 1j * u + d) * T
        - 2 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = ((kappa - rho * sigma_v * 1j * u + d) / sigma_v**2) * (
        (1 - exp_dT) / (1 - g * exp_dT)
    )
    return np.exp(1j * u * x + C + D * v0)


def _prices_batch(S, Ks, T, r, v0, kappa, theta, sigma_v, rho):
    us     = _QUAD_U
    cf_u   = heston_cf(us,      S, T, r, v0, kappa, theta, sigma_v, rho)
    cf_u_i = heston_cf(us - 1j, S, T, r, v0, kappa, theta, sigma_v, rho)

    Ks     = np.asarray(Ks, dtype=float)
    log_Ks = np.log(Ks)
    exp_term = np.exp(-1j * us[:, None] * log_Ks[None, :])
    kernel   = exp_term * (cf_u_i[:, None] - Ks[None, :] * cf_u[:, None]) / (1j * us[:, None])
    integrals = np.trapezoid(np.real(kernel), us, axis=0)

    return 0.5 * (S - Ks * np.exp(-r * T)) + np.exp(-r * T) * integrals / np.pi


def heston_price(S, K, T, r, v0, kappa, theta, sigma_v, rho):
    return float(_prices_batch(S, [K], T, r, v0, kappa, theta, sigma_v, rho)[0])


def calibrate_heston(market_prices, strikes, maturities, S, r):
    market_prices = np.array(market_prices)
    strikes       = np.array(strikes, dtype=float)
    maturities    = np.array(maturities, dtype=float)
    unique_Ts     = np.unique(maturities)

    def objective(params):
        v0, kappa, theta, sigma_v, rho = params
        model_prices = np.empty_like(market_prices)
        for T in unique_Ts:
            mask = maturities == T
            try:
                model_prices[mask] = _prices_batch(S, strikes[mask], T, r,
                                                   v0, kappa, theta, sigma_v, rho)
            except Exception:
                model_prices[mask] = 1e6
        return float(np.sum((model_prices - market_prices) ** 2))

    bounds = [(1e-3, 0.5), (0.1, 10.0), (1e-3, 0.5), (0.01, 2.0), (-0.99, 0.99)]
    x0     = [0.04, 2.0, 0.04, 0.3, -0.7]
    result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B",
                      options={"maxiter": 1000, "ftol": 1e-6, "gtol": 1e-5})
    v0, kappa, theta, sigma_v, rho = result.x
    return {
        "v0": v0, "kappa": kappa, "theta": theta,
        "sigma_v": sigma_v, "rho": rho,
        "success": result.success, "rmse": np.sqrt(result.fun / len(strikes))
    }
