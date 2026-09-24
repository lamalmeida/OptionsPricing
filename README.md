# Options Pricing

Small Python implementations of two standard option-pricing approaches:

- Black-Scholes-Merton pricing for European calls and puts, including Greeks and implied volatility.
- A binomial tree pricer supporting early exercise, suitable for American-style calls and puts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run the tests

```bash
pytest
```

## Files

- `european_pricer.py` — Black-Scholes-Merton pricing, Greeks, and implied volatility.
- `binomial_pricer.py` — recombining binomial-tree option pricer.
- `test_european_pricer.py` — pricing, Greeks, parity, edge-case, and implied-volatility tests.
- `test_binomial_pricer.py` — pricing, convergence, exercise, and validation tests.
