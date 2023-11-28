import numpy as np


class Sigmoid:
    """Applies the sigmoid activation function"""

    def forward(self, data: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-data))
