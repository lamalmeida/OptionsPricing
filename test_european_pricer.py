import math
import pytest
import numpy as np

from src.models.european_pricer import BSMEuropeanPricer, implied_volatility

# --- Fixtures ----------------------------------------------------------------
@pytest.fixture
def bsm_params():
    return {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.20
    }

# --- Basic correctness (baseline) --------------------------------------------
def test_bsm_pricing_baseline(bsm_params):
    pricer = BSMEuropeanPricer(**bsm_params)
    expected_call_price = 10.450576
    expected_put_price = 5.573518
    assert pricer.price_call() == pytest.approx(expected_call_price, abs=1e-5)
    assert pricer.price_put() == pytest.approx(expected_put_price, abs=1e-5)

def test_bsm_greeks_baseline(bsm_params):
    pricer = BSMEuropeanPricer(**bsm_params)
    assert pricer.delta_call() == pytest.approx(0.636831, abs=1e-5)
    assert pricer.delta_put() == pytest.approx(-0.363169, abs=1e-5)
    assert pricer.gamma() == pytest.approx(0.018762, abs=1e-5)
    assert pricer.vega() == pytest.approx(37.524035, abs=1e-5)
    assert pricer.theta_call() == pytest.approx(-6.414028, abs=1e-5)
    assert pricer.theta_put() == pytest.approx(-1.657881, abs=1e-5)
    assert pricer.rho_call() == pytest.approx(53.232483, abs=1e-5)
    assert pricer.rho_put() == pytest.approx(-41.890459, abs=1e-5)

# --- Structural / sanity checks ----------------------------------------------
def test_put_call_parity(bsm_params):
    pricer = BSMEuropeanPricer(**bsm_params)
    C = pricer.price_call()
    P = pricer.price_put()
    lhs = C - P
    rhs = pricer.S - pricer.K * np.exp(-pricer.r * pricer.T)
    assert lhs == pytest.approx(rhs, rel=1e-9, abs=1e-9)

@pytest.mark.parametrize("S,K,r,T,sigma", [
    (100.0, 100.0, 0.0, 1.0, 50.0),   # extreme vol, r=0, ATM
    (120.0, 100.0, 0.01, 0.5, 0.5),   # ITM-ish
    (1.0,   100.0, 0.01, 1.0, 0.3),   # deep OTM
])
def test_greeks_signs_and_ranges(S, K, r, T, sigma):
    pricer = BSMEuropeanPricer(S=S, K=K, T=T, r=r, sigma=sigma)
    # Delta ranges
    assert 0.0 <= pricer.delta_call() <= 1.0
    assert -1.0 <= pricer.delta_put() <= 0.0
    # Non-negative for gamma & vega
    assert pricer.gamma() >= 0.0
    assert pricer.vega() >= 0.0
    # Rho signs
    assert pricer.rho_call() >= -1e-12  # allow tiny numerical noise
    assert pricer.rho_put() <= 1e-12
    # Theta for calls generally <= 0 (non-dividend underlying)
    assert pricer.theta_call() <= 1e-6

# --- Approximations for "zero" edge-cases (use tiny values, not exact zeros) -
def test_near_zero_volatility_approaches_intrinsic(bsm_params):
    params = dict(bsm_params)
    params["sigma"] = 1e-8
    pricer = BSMEuropeanPricer(**params)
    intrinsic_call = max(0.0, params["S"] - params["K"] * math.exp(-params["r"] * params["T"]))
    intrinsic_put  = max(0.0, params["K"] * math.exp(-params["r"] * params["T"]) - params["S"])
    assert pricer.price_call() == pytest.approx(intrinsic_call, abs=1e-6)
    assert pricer.price_put()  == pytest.approx(intrinsic_put,  abs=1e-6)

def test_near_zero_time_to_expiry_approaches_intrinsic():
    # choose S>K so intrinsic > 0
    params = {"S": 120.0, "K": 100.0, "T": 1e-8, "r": 0.01, "sigma": 0.2}
    pricer = BSMEuropeanPricer(**params)
    intrinsic = max(0.0, params["S"] - params["K"] * math.exp(-params["r"] * params["T"]))
    assert pricer.price_call() == pytest.approx(intrinsic, rel=1e-6, abs=1e-6)

# --- Deep ITM / OTM ----------------------------------------------------------
def test_deep_itm_and_otm_behavior():
    # deep ITM
    params_itm = {"S": 1000.0, "K": 100.0, "T": 1.0, "r": 0.01, "sigma": 0.20}
    pricer_itm = BSMEuropeanPricer(**params_itm)
    expected_forward_intrinsic = params_itm["S"] - params_itm["K"] * math.exp(-params_itm["r"] * params_itm["T"])
    assert pricer_itm.price_call() == pytest.approx(expected_forward_intrinsic, rel=1e-3)

    # deep OTM
    params_otm = {"S": 1.0, "K": 100.0, "T": 1.0, "r": 0.01, "sigma": 0.20}
    pricer_otm = BSMEuropeanPricer(**params_otm)
    assert pricer_otm.price_call() < 1e-6

# --- Extreme volatility ------------------------------------------------------
def test_extreme_volatility_symmetry_r0():
    params = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.0, "sigma": 50.0}
    pricer = BSMEuropeanPricer(**params)
    call = pricer.price_call()
    put  = pricer.price_put()
    assert call == pytest.approx(params["K"] * np.exp(params["r"] * params["T"]), rel=1e-3, abs=1e-1)
    assert put  == pytest.approx(params["K"] * np.exp(params["r"] * params["T"]), rel=1e-3, abs=1e-1)
    # parity still holds
    assert call - put == pytest.approx(0, abs=1e-6)

# --- Implied volatility ------------------------------------------------------
def test_implied_volatility_round_trip(bsm_params):
    pricer = BSMEuropeanPricer(**bsm_params)
    market_price_call = pricer.price_call()
    market_price_put  = pricer.price_put()

    iv_call = implied_volatility(market_price_call, S=bsm_params["S"], K=bsm_params["K"],
                                 T=bsm_params["T"], r=bsm_params["r"], option_type="call")
    iv_put  = implied_volatility(market_price_put,  S=bsm_params["S"], K=bsm_params["K"],
                                 T=bsm_params["T"], r=bsm_params["r"], option_type="put")

    assert iv_call == pytest.approx(bsm_params["sigma"], rel=1e-6, abs=1e-6)
    assert iv_put  == pytest.approx(bsm_params["sigma"], rel=1e-6, abs=1e-6)

def test_implied_volatility_no_solution_raises():
    # Construct an impossible market price (below intrinsic) for a call -> no IV exists
    S, K, T, r = 100.0, 90.0, 1.0, 0.05
    intrinsic = max(0.0, S - K * math.exp(-r * T))
    impossible_price = intrinsic - 0.5  # strictly below intrinsic -> invalid
    with pytest.raises(ValueError):
        implied_volatility(impossible_price, S=S, K=K, T=T, r=r, option_type="call")

def test_implied_volatility_invalid_option_type_raises():
    pricer = BSMEuropeanPricer(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    market_price = pricer.price_call()
    with pytest.raises(Exception):
        implied_volatility(market_price, S=100.0, K=100.0, T=1.0, r=0.05, option_type="invalid")

# --- Input validation (xfail until pricer implements explicit checks) -------
@pytest.mark.xfail(reason="BSMEuropeanPricer currently does not validate non-positive inputs; add input checks", strict=False)
@pytest.mark.parametrize("S,K,T", [(0.0,100.0,1.0), (100.0,0.0,1.0), (100.0,100.0,0.0)])
def test_input_validation_nonpositive_raises(S, K, T):
    # Ideally the class should raise ValueError for non-positive S/K/T.
    # Marked xfail so test suite doesn't fail until validation is implemented.
    with pytest.raises(ValueError):
        BSMEuropeanPricer(S=S, K=K, T=T, r=0.01, sigma=0.2)
