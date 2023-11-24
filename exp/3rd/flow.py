import pandas as pd
from module.context import Atmacup15Context
from module.cross_validation import UnknownUserKFold
from module.lgb.experiment import Atmacup15Experiment, Atmacup15ExperimentConfig
from module.metrics import RMSE
from module.preprocess import Atmacup15Preprocess
from ocha.config.config import GlobalConfig
from ocha.dataset.cross_validator import CrossValidator
from ocha.experiment.results import ExperimentResult, RemakeResult
from prefect import flow, tags, task

conf = GlobalConfig(version=1, n_fold=5, seed=1013)

train = pd.read_csv("../../input/train.csv")
test = pd.read_csv("../../input/test.csv")
anime = pd.read_csv("../../input/anime.csv")
sample_submission = pd.read_csv("../../input/sample_submission.csv")


@task(name="load data", log_prints=True)
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    # make train_test_anime
    test["score"] = 0
    train_test = pd.concat([train, test], axis=0).reset_index(drop=True)
    train_test_anime = train_test.merge(anime, on="anime_id", how="left").reset_index(drop=True)

    print("train_test_anime:", train_test_anime.head())

    # sample oof
    sample_oof = pd.DataFrame()
    sample_oof[["user_id", "anime_id", "score"]] = train_test_anime.iloc[: len(train)][["user_id", "anime_id", "score"]]

    print("sample oof:", sample_oof.head())

    return train_test_anime, sample_oof


@task(name="preprocess", log_prints=True)
def preprocess(train_test_anime: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    preprocess = Atmacup15Preprocess(source=train_test_anime)
    preprocess.preprocess()
    preprocess.get_procesed().head()
    train_processed = preprocess.get_procesed()[: len(train)].reset_index(drop=True)
    test_processed = preprocess.get_procesed()[len(train) :].reset_index(drop=True)

    return train_processed, test_processed


@task(name="get context", log_prints=True)
def get_context(
    train: pd.DataFrame, test: pd.DataFrame, sample_oof: pd.DataFrame, sample_submission: pd.DataFrame
) -> Atmacup15Context:
    return Atmacup15Context(train=train, test=test, sample_oof_df=sample_oof, sample_submission_df=sample_submission)


@task(name="get cross validator", log_prints=True)
def get_cross_validator(n_splits_cv: int, n_splits_uu: int) -> CrossValidator:
    fold_df = pd.DataFrame()
    fold_df["fold"] = [-1 for _ in range(len(train))]
    cv = UnknownUserKFold(n_splits_cv, n_splits_uu)
    for fold, (_, valid_idx) in enumerate(cv.split(train, groups=train["user_id"])):
        fold_df.loc[valid_idx, "fold"] = fold

    assert len(fold_df[fold_df["fold"] == -1]) == 0

    print("fold:", fold_df.head())

    return CrossValidator(folds=fold_df)


@task(name="experiment", log_prints=True)
def experiment(
    context: Atmacup15Context, cross_validator: CrossValidator, folds: list[int] | None = None
) -> ExperimentResult | RemakeResult:
    exp_conf = Atmacup15ExperimentConfig(
        version=conf.version,
        n_fold=conf.n_fold,
        seed=conf.seed,
        scoring=RMSE(),
        cross_validator=cross_validator,
        folds=folds,
    )

    exp = Atmacup15Experiment(context=context, config=exp_conf)

    result = exp.run()

    print(result.submission_df.head())

    return result


@flow(name=f"atmacup-15-v{conf.version}")
def run() -> None:
    train_test_anime, sample_oof = load()
    train_processed, test_processed = preprocess(train_test_anime)
    context = get_context(train_processed, test_processed, sample_oof, sample_submission)
    cross_validator = get_cross_validator(n_splits_cv=5, n_splits_uu=18)
    experiment(context, cross_validator, folds=None)


if __name__ == "__main__":
    with tags("competition:atmacup-15"):
        run()
