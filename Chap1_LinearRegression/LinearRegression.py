import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, alpha=0.05, n_iters=2000, x=None, y=None, theta0=1, theta1=1):
        self.alpha = alpha
        self.n_iters = n_iters
        self.x = x
        self.y = y
        self.theta0 = theta0
        self.theta1 = theta1

    def _calculate_cost_function(self, x, y, theta0, theta1):
        m = len(x)
        cost = 0
        for i in range(m):
            cost += (x[i] * theta1 + theta0 - y[i]) ** 2
        return 1 / 2 / m * cost

    def _gradient_descent(self, x, y, theta0, theta1, n_iters):
        costs, m = [0 for _ in range(n_iters)], len(x)
        for j in range(n_iters):
            sum_gradient0, sum_gradient1 = 0, 0
            for i in range(m):
                sum_gradient0 += (x[i] * theta1 + theta0 - y[i])
                sum_gradient1 += (x[i] * theta1 + theta0 - y[i]) * x[i]
            theta0 = theta0 - self.alpha / m * sum_gradient0
            theta1 = theta1 - self.alpha / m * sum_gradient1
            costs.append(self._calculate_cost_function(x, y, theta0, theta1))
        return costs[-1], theta0, theta1

    def _standardization(self, x):
        mu = np.mean(x)
        sigma = np.std(x)
        standardized_x = (x - mu) / sigma
        return standardized_x

    def _de_standardization(self, theta0, theta1, x):
        mu = np.mean(x)
        sigma = np.std(x)
        theta0 = theta0 - theta1 * mu / sigma
        theta1 = theta1 / sigma
        return theta0, theta1

    def fit(self, x, y, theta0, theta1, n_iters):
        # fit function is supposed to return y_hat
        standardized_x = self._standardization(x)
        cost, theta0, theta1 = self._gradient_descent(x=standardized_x, y=y,
                                                     theta0=theta0, theta1=theta1,
                                                     n_iters=n_iters)
        theta0, theta1 = self._de_standardization(theta0, theta1, x)
        y_hat = theta1 * x + theta0
        return y_hat


if __name__ == "__main__":
    x = np.array([68, 60, 51, 43, 31])
    y = np.array([37.49, 36.46, 67.28, 93.75, 140.22])
    t = LinearRegression(alpha=0.003, x=x, y=y)
    y_hat = t.fit(x, y, 1, 1, n_iters=2000)
    plt.plot(x, y_hat)
    plt.scatter(x, y)