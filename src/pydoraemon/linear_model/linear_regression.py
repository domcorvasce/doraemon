import numpy as np


class LinearRegression:
    """Implements a linear regression model trainable via Gradient Descent"""

    def __init__(self, features: int):
        # TODO: Remove need for "features" and compute weights on `train` instead
        self._weights = np.zeros(features)
        self._bias = 0.0

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Computes predictions for the set of given datapoints"""
        return data @ self._weights + self._bias

    def mse(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Computes the mean squared error of the model given a set of datapoints and their labels"""
        return ((self.predict(data) - labels) ** 2).sum() / (0.5 * len(data))

    def _converged(self, current_error: float, previous_error: float | None) -> bool:
        """Checks if Gradient Descent has converged

        i.e. the absolute difference between the MSE of the last two epochs is infinitesimaly small
        """
        return (
            previous_error is not None
            and abs(current_error - previous_error) < 0.0000001
        )

    def train(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        learning_rate: float = 0.1,
        max_epochs: int = 1000,
    ) -> None:
        """Train the model on the set of given datapoints and associated labels"""
        last_epoch_error = None
        epochs_left = max_epochs

        while epochs_left > 0:
            epoch_error = self.mse(data, labels)
            if self._converged(epoch_error, last_epoch_error):
                break

            last_epoch_error = epoch_error

            residuals = (self.predict(data) - labels) * (1 / len(data))
            self._bias -= learning_rate * residuals.sum()
            self._weights -= learning_rate * np.array(
                # Compute the derivative of the cost function w.r.t. each weight
                [(data[:, i] * residuals).sum() for i in range(len(self._weights))]
            )

            epochs_left -= 1
