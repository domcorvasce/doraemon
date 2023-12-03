import numpy as np


class ReLU:
    """Applies the Rectified Linear Unit (ReLU) activation function"""

    def forward(self, data: np.ndarray) -> np.ndarray:
        return np.clip(data, 0, None)
