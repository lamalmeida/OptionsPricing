import numpy as np

class BinomialPricer:
    def __init__(self, S, K, T, r, sigma, N):
        if S <= 0:
            raise ValueError("Underlying asset price (S) must be positive.")
        if K <= 0:
            raise ValueError("Strike price (K) must be positive.")
        if T <= 0:
            raise ValueError("Time to expiration (T) cannot be negative.")
        if sigma <= 0:
            raise ValueError("Volatility (sigma) must be positive.")
        if N <= 0:
            raise ValueError("Number of steps (N) must be a positive integer.")

        self.S = S          # Underlying asset price
        self.K = K          # Strike price
        self.T = T          # Time to expiration (in years)
        self.r = r          # Risk-free interest rate
        self.sigma = sigma  # Volatility
        self.N = N          # Number of steps

        self._calculate_tree_params()

    def _calculate_tree_params(self):
        self.delta_t = self.T / self.N
        self.u = np.exp(self.sigma * np.sqrt(self.delta_t))
        self.d = 1.0 / self.u
        self.p = (np.exp(self.r * self.delta_t) - self.d) / (self.u - self.d)

    def _build_option_tree(self, option_type):
        option_tree = [[] for _ in range(self.N + 1)]

        for i in range(self.N + 1):
            stock_price_at_node = self.S * (self.u ** i) * (self.d ** (self.N - i))
            if option_type == "call":
                payoff = max(0.0, stock_price_at_node - self.K)
            elif option_type == "put":
                payoff = max(0.0, self.K - stock_price_at_node)
            option_tree[self.N].append(payoff)
        
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                option_up = option_tree[i + 1][j + 1]
                option_down = option_tree[i + 1][j]
                continuation_value = np.exp(-self.r * self.delta_t) * (self.p * option_up + (1 - self.p) * option_down)

                stock_price_at_node = self.S * (self.u ** j) * (self.d ** (i - j))
                if option_type == "call":
                    exercise_value = max(0.0, stock_price_at_node - self.K)
                elif option_type == "put":
                    exercise_value = max(0.0, self.K - stock_price_at_node)
                option_tree[i].append(max(continuation_value, exercise_value))
        
        return option_tree[0][0]

    def price_call(self):
        return self._build_option_tree(option_type="call")

    def price_put(self):
        return self._build_option_tree(option_type="put")