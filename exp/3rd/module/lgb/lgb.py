import pickle
from typing import Any

import lightgbm as lgb
from lightgbm import Booster
from numpy import ndarray
from ocha.models.model_config import ModelConfig
from ocha.models.model_wrapper import FitResult, ModelWrapper
from pandas import DataFrame, Series

from ..metrics import RMSE


class LGBMRegressionConfig(ModelConfig):
    save_dir: str
    save_file_name: str
    model_file_type: str
    seed: int
    categorical_features: list[str]
    boosting_type: str = "gbdt"
    objective: str = "regression"
    metric: str = "rmse"
    num_boost_round: int = 10000
    learning_rate: float = 0.01
    early_stopping_rounds: int = 500
    lambda_l1: float = 0.1
    lambda_l2: float = 0.1
    max_depth: int = 7
    feature_fraction: float = 1.0
    num_leaves: int = 64
    min_data_in_leaf: int = 100
    max_bin: int = 255
    verbose_eval: int = 100
    verbosity: int = -1
    num_threads: int = -1
    is_debug: bool = False

    @property
    def model_params(self) -> dict:
        return {
            "seed": self.seed,
            "boosting_type": self.boosting_type,
            "objective": self.objective,
            "metric": self.metric,
            "num_boost_round": self.num_boost_round,
            "early_stopping_round": self.early_stopping_rounds,
            "learning_rate": self.learning_rate,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "max_depth": self.max_depth,
            "feature_fraction": self.feature_fraction,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "max_bin": self.max_bin,
            "verbosity": self.verbosity,
            "num_threads": self.num_threads,
        }


class LGBMRegression(ModelWrapper):
    config: LGBMRegressionConfig
    scoring: RMSE
    model: Booster | None = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.config.is_debug:
            print("This is Debug Mode.")
            self.config.num_boost_round = 1

    def build(self) -> None:
        pass

    def load(self):
        with open(self.config.save_model_path, "rb") as f:
            self.model = pickle.load(f)

    def fit(
        self,
        X_train: DataFrame | ndarray,
        y_train: Series | ndarray,
        X_valid: DataFrame | ndarray,
        y_valid: Series | ndarray,
    ) -> FitResult:
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

        self.model = lgb.train(
            self.config.model_params,
            lgb_train,
            categorical_feature=self.config.categorical_features,
            valid_names=["train", "valid"],
            valid_sets=[lgb_train, lgb_valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds, verbose=True),
                lgb.log_evaluation(self.config.verbose_eval),
            ],
        )

        oof_prediction = self.model.predict(X_valid, num_iteration=self.model.best_iteration)
        score = self.scoring.execute(y_valid, oof_prediction)

        self.save()

        return FitResult(
            model=self,
            oof_prediction=oof_prediction,
            score=score,
        )

    def predict(self, X_test: DataFrame) -> ndarray:
        return self.model.predict(X_test, num_iteration=self.model.best_iteration)

    def save(self):
        with open(self.config.save_model_path, "wb") as f:
            pickle.dump(self.model, f)
