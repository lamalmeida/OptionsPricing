import math
import pytest
import numpy as np

from binomial_pricer import BinomialPricer

# --- Fixtures ----------------------------------------------------------------
@pytest.fixture
def binomial_params():
    return {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.20,
        "N": 200   # reasonably high for convergence
    }

# --- Basic correctness (baseline) --------------------------------------------
def test_binomial_pricing_baseline(binomial_params):
    pricer = BinomialPricer(**binomial_params)
    call_price = pricer.price_call()
    put_price  = pricer.price_put()
    # For large N, American call (no dividends) ~ European call
    expected_call = 10.450576
    assert call_price == pytest.approx(expected_call, rel=1e-2, abs=1e-2)
    # American put >= European put
    european_put = 5.573518
    assert put_price >= european_put - 1e-6

# --- Structural / sanity checks ----------------------------------------------
def test_american_call_equals_european_when_no_dividends(binomial_params):
    pricer = BinomialPricer(**binomial_params)
    assert pricer.price_call() == pytest.approx(10.45, rel=1e-2)

def test_american_put_greater_than_or_equal_european(binomial_params):
    pricer = BinomialPricer(**binomial_params)
    european_put = 5.573518
    assert pricer.price_put() >= european_put - 1e-12

def test_increasing_steps_converges(binomial_params):
    N_values = [10, 50, 100, 500]
    call_prices = []
    for N in N_values:
        params = dict(binomial_params)
        params["N"] = N
        call_prices.append(BinomialPricer(**params).price_call())
    # Prices should stabilize as N grows
    diffs = np.diff(call_prices)
    assert all(abs(d) < 0.05 for d in diffs[1:])  # convergence behavior

# --- Edge cases ---------------------------------------------------------------
def test_near_zero_volatility_approaches_intrinsic(binomial_params):
    params = dict(binomial_params)
    params["sigma"] = 1e-4
    pricer = BinomialPricer(**params)
    intrinsic_call = max(0.0, params["S"] - params["K"])
    intrinsic_put  = max(0.0, params["K"] - params["S"])
    assert pricer.price_call() == pytest.approx(intrinsic_call, abs=1e-6)
    assert pricer.price_put()  == pytest.approx(intrinsic_put,  abs=1e-6)

def test_near_zero_time_to_expiry_approaches_intrinsic(binomial_params):
    params = dict(binomial_params)
    params["T"] = 1e-10
    params["N"] = 1  
    pricer = BinomialPricer(**params)
    intrinsic_call = max(0.0, params["S"] - params["K"])
    intrinsic_put  = max(0.0, params["K"] - params["S"])
    assert pricer.price_call() == pytest.approx(intrinsic_call, abs=1e-3)
    assert pricer.price_put()  == pytest.approx(intrinsic_put,  abs=1e-3)

def test_deep_in_the_money_put_exercise(binomial_params):
    params = dict(binomial_params)
    params["S"] = 10.0
    params["K"] = 100.0
    pricer = BinomialPricer(**params)
    # American put should ~ immediate exercise (K-S discounted)
    expected = params["K"] - params["S"]
    assert pricer.price_put() == pytest.approx(expected, rel=1e-2)

def test_deep_in_the_money_call(binomial_params):
    params = dict(binomial_params)
    params["S"] = 200.0
    params["K"] = 100.0
    pricer = BinomialPricer(**params)
    # Call ~ S - K * exp(-rT) for large ITM
    expected = params["S"] - params["K"] * math.exp(-params["r"] * params["T"])
    assert pricer.price_call() == pytest.approx(expected, rel=1e-2)

def test_deep_out_of_the_money(binomial_params):
    params = dict(binomial_params)
    params["S"] = 1.0
    params["K"] = 100.0
    pricer = BinomialPricer(**params)
    assert pricer.price_call() < 1e-6
    assert pricer.price_put() > 90.0  # basically intrinsic

# --- Extreme volatility ------------------------------------------------------
def test_extreme_volatility(binomial_params):
    params = dict(binomial_params)
    params["sigma"] = 5.0
    pricer = BinomialPricer(**params)
    call = pricer.price_call()
    put  = pricer.price_put()
    # Prices shouldn't blow up; bounded by S and K
    assert 0.0 <= call <= params["S"]
    assert 0.0 <= put <= params["K"]

# --- Input validation --------------------------------------------------------
@pytest.mark.parametrize("S,K,T,sigma,N", [
    (0.0, 100.0, 1.0, 0.2, 50),   # invalid S
    (100.0, 0.0, 1.0, 0.2, 50),   # invalid K
    (100.0, 100.0, 0.0, 0.2, 50), # invalid T
    (100.0, 100.0, 1.0, 0.0, 50), # invalid sigma
    (100.0, 100.0, 1.0, 0.2, 0),  # invalid N
])
def test_invalid_inputs_raise(S, K, T, sigma, N):
    with pytest.raises(ValueError):
        BinomialPricer(S=S, K=K, T=T, r=0.01, sigma=sigma, N=N)
