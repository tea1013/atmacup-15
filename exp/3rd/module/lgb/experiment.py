import os
from typing import Any, Union

import numpy as np
from numpy import ndarray
from ocha.dataset.cross_validator import CrossValidator
from ocha.experiment.experiment import Experiment, ExperimentConfig
from ocha.experiment.results import ExperimentResult, RemakeResult, TestResult, TrainResult, ValidResult
from ocha.util.timer import Timer
from pandas import DataFrame

from ..context import Atmacup15Context
from ..dataset import Atmacup15Dataset
from ..metrics import RMSE
from .lgb import LGBMRegression, LGBMRegressionConfig


class Atmacup15ExperimentConfig(ExperimentConfig):
    version: int
    n_fold: int
    seed: int
    scoring: RMSE
    cross_validator: CrossValidator
    folds: list[int] | None = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        os.makedirs("./oof/lgb", exist_ok=True)
        os.makedirs("./models/lgb", exist_ok=True)
        os.makedirs("./submission/lgb", exist_ok=True)


class Atmacup15Experiment(Experiment):
    context: Atmacup15Context
    config: Atmacup15ExperimentConfig

    def build_conf(self, fold: int, categorical_features: list[str]) -> LGBMRegressionConfig:
        return LGBMRegressionConfig(
            save_dir="lgb",
            save_file_name=f"fold-{fold}",
            model_file_type="pickle",
            seed=self.config.seed,
            categorical_features=categorical_features,
            num_boost_round=10000,
            max_bin=500,
            verbose_eval=500,
            is_debug=True,
        )

    def build_model(self, conf: LGBMRegressionConfig) -> LGBMRegression:
        return LGBMRegression(config=conf, scoring=self.config.scoring)

    def run(self) -> ExperimentResult:
        timer = Timer()
        timer.start()

        train_result = self.train()

        print("Saveing oof ...")
        oof_df = self.save_oof(train_result.oof_prediction, train_result.score)
        print("done.")

        print("Prediction ...")
        test_result = self.test()
        print("done.")

        print("Saving submission_df ...")
        submission_df = self.save_submission(test_result.test_prediction, train_result.score)

        timer.end()

        print(f"Experiment End. [score: {train_result.score}, time: {timer.result}]")

        return ExperimentResult(
            fit_results=train_result.fit_results,
            oof_prediction=train_result.oof_prediction,
            test_prediction=test_result.test_prediction,
            oof_df=oof_df,
            submission_df=submission_df,
            score=train_result.score,
            time=timer.result,
        )

    def train(self) -> TrainResult:
        fit_results = []
        oof_prediction = np.zeros(len(self.context.sample_oof_df))
        for fold in self.get_folds():
            train_idx, valid_idx = self.config.cross_validator.fold_index(fold=fold)
            train = self.context.train.iloc[train_idx].reset_index(drop=True)
            valid = self.context.train.iloc[valid_idx].reset_index(drop=True)
            dataset = Atmacup15Dataset(train=train, valid=valid, test=self.context.test)
            dataset.processing_train()
            dataset.processing_valid()

            conf = self.build_conf(fold=fold, categorical_features=dataset.categorical_features)
            model = self.build_model(conf=conf)
            result = model.fit(dataset.train_X, dataset.train_y, dataset.valid_X, dataset.valid_y)

            fit_results.append(result)
            oof_prediction[valid_idx] = result.oof_prediction

        score = self.config.scoring.execute(self.context.train["score"].values, oof_prediction)

        return TrainResult(fit_results=fit_results, oof_prediction=oof_prediction, score=score)

    def valid(self) -> ValidResult:
        oof_prediction = np.zeros(len(self.context.train))
        for fold in self.get_folds():
            train_idx, valid_idx = self.config.cross_validator.fold_index(fold=fold)
            train = self.context.train.iloc[train_idx].reset_index(drop=True)
            valid = self.context.train.iloc[valid_idx].reset_index(drop=True)
            dataset = Atmacup15Dataset(train=train, valid=valid, test=self.context.test)
            dataset.processing_train()
            dataset.processing_valid()

            conf = self.build_conf(fold=fold, categorical_features=dataset.categorical_features)
            model = self.build_model(conf=conf)
            model.load()

            oof_prediction[valid_idx] = model.predict(dataset.valid_X)

        score = self.config.scoring.execute(self.context.train["score"].values, oof_prediction)

        return ValidResult(oof_prediction=oof_prediction, score=score)

    def test(self) -> TestResult:
        predictions = []
        for fold in self.get_folds():
            train_idx, valid_idx = self.config.cross_validator.fold_index(fold=fold)
            train = self.context.train.iloc[train_idx].reset_index(drop=True)
            valid = self.context.train.iloc[valid_idx].reset_index(drop=True)
            dataset = Atmacup15Dataset(train=train, valid=valid, test=self.context.test)
            dataset.processing_train()
            dataset.processing_test()

            conf = self.build_conf(fold=fold, categorical_features=dataset.categorical_features)
            model = self.build_model(conf=conf)
            model.load()

            prediction = model.predict(dataset.test_X)
            predictions.append(prediction)

        test_prediction = np.mean(predictions, axis=0)

        return TestResult(test_prediction=test_prediction)

    def test_seq(self, test_X: Union[DataFrame, ndarray]) -> TestResult:
        pass

    def optimize(self) -> None:
        pass

    def save_oof(self, oof_prediction: ndarray, score: float) -> DataFrame:
        oof_df = self.context.make_oof(oof_prediction)
        oof_df.to_csv(
            f"./oof/lgb/v{self.config.version}_oof_cv{score:.4f}.csv",
            index=False,
        )

        return oof_df

    def save_submission(self, test_prediction: ndarray, score: float) -> DataFrame:
        submission_df = self.context.make_submission(test_prediction)
        submission_df.to_csv(
            f"./submission/lgb/v{self.config.version}_submission_cv{score:.4f}.csv",
            index=False,
        )

        return submission_df

    def remake(self) -> RemakeResult:
        valid_result = self.valid()
        test_result = self.test()

        oof_df = self.save_oof(valid_result.oof_prediction, valid_result.score)
        submission_df = self.save_submission(test_result.test_prediction, valid_result.score)

        return RemakeResult(oof_df=oof_df, submission_df=submission_df, score=valid_result.score)
