import numpy as np
from sklearn.model_selection import GroupKFold, KFold


class UnknownUserKFold:
    def __init__(self, n_splits_cv: int, n_splits_uu: int):
        self.n_splits_cv = n_splits_cv
        self.n_splits_uu = n_splits_uu

    def split(self, X, y=None, groups=None):
        splits_cv = KFold(n_splits=self.n_splits_cv, shuffle=True, random_state=0).split(X, y)
        splits_uu = GroupKFold(n_splits=self.n_splits_uu).split(X, groups=groups)
        for _ in range(self.n_splits_cv):
            train_index, test_index = next(splits_cv)
            _, uu_index = next(splits_uu)
            train_index = np.setdiff1d(train_index, uu_index)
            test_index = np.union1d(test_index, uu_index)

            yield train_index, test_index
