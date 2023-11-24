import numpy as np
from numpy import ndarray
from ocha.models.metrics import Metrics
from sklearn.metrics import mean_squared_error


class RMSE(Metrics):
    name: str = "RMSE"

    def execute(self, y_true: ndarray, y_pred: ndarray) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred))
