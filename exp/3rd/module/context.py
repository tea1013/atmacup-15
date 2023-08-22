import numpy as np
from numpy import ndarray
from ocha.experiment.context import Context
from pandas import DataFrame


class Atmacup15Context(Context):
    def __init__(
        self, train: DataFrame, test: DataFrame | None, sample_oof_df: DataFrame, sample_submission_df: DataFrame
    ) -> None:
        super().__init__(train, test, sample_oof_df, sample_submission_df)

    def make_oof(self, oof_prediction: ndarray) -> DataFrame:
        oof_df = self.sample_oof_df.copy()
        oof_df["pred"] = self.post_process(oof_prediction)

        return oof_df

    def make_submission(self, test_prediction: ndarray) -> DataFrame:
        submission_df = self.sample_submission_df.copy()
        submission_df["score"] = self.post_process(test_prediction)

        return submission_df

    def post_process(self, prediction: ndarray) -> ndarray:
        _prediction = np.where(prediction < 1, 1, prediction)
        _prediction = np.where(_prediction > 10, 10, _prediction)

        return _prediction
