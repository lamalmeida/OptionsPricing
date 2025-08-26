from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np

class BSMEuropeanPricer:
    def __init__(self, S, K, T, r, sigma):
        if S <= 0:
            raise ValueError("Underlying asset price (S) must be positive.")
        if K <= 0:
            raise ValueError("Strike price (K) must be positive.")
        if T <= 0:
            raise ValueError("Time to expiration (T) cannot be negative.")
        if sigma <= 0:
            raise ValueError("Volatility (sigma) must be positive.")

        self.S = S          # Underlying asset price
        self.K = K          # Strike price
        self.T = T          # Time to expiration (in years)
        self.r = r          # Risk-free interest rate
        self.sigma = sigma  # Volatility

        self._calculate_d1_d2()

    def _calculate_d1_d2(self):
        self.d1 = (
            (np.log(self.S/self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / 
            (self.sigma * np.sqrt(self.T))
        )
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def price_call(self):
        call_price = (
            norm.cdf(self.d1, 0.0, 1.0) * self.S 
            - norm.cdf(self.d2, 0.0, 1.0) * self.K * np.exp(-self.r * self.T)
            )
        return call_price

    def price_put(self):
        put_price = (
            norm.cdf(-self.d2, 0.0, 1.0) * self.K * np.exp(-self.r * self.T)
            - norm.cdf(-self.d1, 0.0, 1.0) * self.S
        )
        return put_price

    def delta_call(self):
        return norm.cdf(self.d1)

    def delta_put(self):
        return norm.cdf(self.d1) - 1

    def gamma(self):
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self):
        return self.S * norm.pdf(self.d1) * np.sqrt(self.T)

    def theta_call(self):
        return (
            - (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T)) - 
            self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        )

    def theta_put(self):
        return (
            - (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T)) + 
            self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        )

    def rho_call(self):
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)

    def rho_put(self):
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)

def implied_volatility(market_price, S, K, T, r, option_type):
    def objective(sigma):
        pricer = BSMEuropeanPricer(S, K, T, r, sigma)
        if option_type == "call":
            return pricer.price_call() - market_price
        elif option_type == "put":
            return pricer.price_put() - market_price
        else: 
            return None
    return brentq(objective, 1e-8, 5.0, maxiter=500, xtol=1e-8)

