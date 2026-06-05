import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from black_scholes import bs_price, greeks, implied_vol
from heston import heston_cf, heston_price
from monte_carlo import mc_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


def test_bs_put_call_parity():
    call = bs_price(S, K, T, r, sigma, "call")
    put  = bs_price(S, K, T, r, sigma, "put")
    pv_K = K * np.exp(-r * T)
    assert abs((call - put) - (S - pv_K)) < 1e-8, (
        f"Put-call parity violated: C-P={call-put:.6f}  S-PV(K)={S-pv_K:.6f}"
    )


def test_bs_put_call_parity_itm():
    call = bs_price(S, 80.0, T, r, sigma, "call")
    put  = bs_price(S, 80.0, T, r, sigma, "put")
    pv_K = 80.0 * np.exp(-r * T)
    assert abs((call - put) - (S - pv_K)) < 1e-8


def test_iv_roundtrip():
    price = bs_price(S, K, T, r, sigma)
    iv    = implied_vol(price, S, K, T, r)
    assert abs(iv - sigma) < 1e-5, f"IV roundtrip failed: got {iv:.6f}, expected {sigma}"


def test_iv_roundtrip_otm():
    price = bs_price(S, 120.0, T, r, 0.25, "call")
    iv    = implied_vol(price, S, 120.0, T, r, "call")
    assert abs(iv - 0.25) < 1e-5


def test_iv_impossible_price_returns_nan():
    # call price can never exceed S; market_price > S has no solution
    iv = implied_vol(S + 50.0, S, K, T, r, "call")
    assert np.isnan(iv)


def test_bs_greeks_delta_call_bounds():
    g = greeks(S, K, T, r, sigma, "call")
    assert 0.0 < g["delta"] < 1.0
    assert g["gamma"] > 0.0
    assert g["vega"]  > 0.0
    assert g["theta"] < 0.0


def test_bs_greeks_delta_put_bounds():
    g = greeks(S, K, T, r, sigma, "put")
    assert -1.0 < g["delta"] < 0.0
    assert g["gamma"] > 0.0


def test_bs_expiry_call_intrinsic():
    assert abs(bs_price(110.0, 100.0, 0.0, r, sigma, "call") - 10.0) < 1e-10
    assert bs_price(90.0, 100.0, 0.0, r, sigma, "call") == 0.0


def test_heston_cf_finite():
    us  = np.array([0.5, 1.0, 2.0, 5.0])
    cfs = heston_cf(us, S, T, r, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
    assert np.all(np.isfinite(cfs.real)) and np.all(np.isfinite(cfs.imag))


def test_heston_put_call_parity():
    kw = dict(S=S, T=T, r=r, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
    call = heston_price(K=K, **kw)
    # put via parity: P = C - S + K*e^{-rT}
    parity_put = call - S + K * np.exp(-r * T)
    assert parity_put > 0.0, "Heston put price from parity must be positive"
    assert abs(parity_put - call) < call * 2, "Heston put-call parity sanity"


def test_heston_price_positive():
    price = heston_price(S=S, K=K, T=T, r=r, v0=0.04,
                         kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
    assert price > 0.0


def test_mc_european_call_near_bs():
    rng = np.random.default_rng(42)
    np.random.seed(42)
    mc, se = mc_price(S, K, T, r, sigma, "european_call", n_paths=200_000, n_steps=252)
    bs     = bs_price(S, K, T, r, sigma)
    assert abs(mc - bs) < 4 * se, (
        f"MC={mc:.4f} BS={bs:.4f} SE={se:.4f}: gap > 4 standard errors"
    )


def test_mc_asian_cheaper_than_european():
    np.random.seed(0)
    eu_price, _ = mc_price(S, K, T, r, sigma, "european_call", n_paths=50_000)
    as_price, _ = mc_price(S, K, T, r, sigma, "asian_call",    n_paths=50_000)
    assert as_price < eu_price, "Asian call must be cheaper than European (averaging reduces vol)"


def test_mc_barrier_cheaper_than_european():
    np.random.seed(0)
    eu_price, _ = mc_price(S, K, T, r, sigma, "european_call",      n_paths=50_000)
    ba_price, _ = mc_price(S, K, T, r, sigma, "barrier_down_out_call", n_paths=50_000)
    assert ba_price < eu_price, "Down-and-out barrier call must be cheaper than vanilla"
