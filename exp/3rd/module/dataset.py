from ocha.dataset.dataset import Dataset
from pandas import DataFrame


class Atmacup15Dataset(Dataset):
    def __init__(self, train: DataFrame, valid: DataFrame, test: DataFrame) -> None:
        super().__init__(train, valid, test)

    @property
    def all_features(self) -> list[str]:
        return (
            [
                "members",
                "watching",
                "completed",
                "on_hold",
                "dropped",
                "plan_to_watch",
                "user_count",
                "user_ip_count",
                "aired_year",
                "duration_min",
                "dropped_rate",
                "completed_rate",
                "watching_rate",
            ]
            + [
                "genre_Action",
                "genre_Adventure",
                "genre_Cars",
                "genre_Comedy",
                "genre_Dementia",
                "genre_Demons",
                "genre_Drama",
                "genre_Ecchi",
                "genre_Fantasy",
                "genre_Game",
                "genre_Harem",
                "genre_Hentai",
                "genre_Historical",
                "genre_Horror",
                "genre_Josei",
                "genre_Kids",
                "genre_Magic",
                "genre_Martial Arts",
                "genre_Mecha",
                "genre_Military",
                "genre_Music",
                "genre_Mystery",
                "genre_Parody",
                "genre_Police",
                "genre_Psychological",
                "genre_Romance",
                "genre_Samurai",
                "genre_School",
                "genre_Sci-Fi",
                "genre_Seinen",
                "genre_Shoujo",
                "genre_Shoujo Ai",
                "genre_Shounen",
                "genre_Shounen Ai",
                "genre_Slice of Life",
                "genre_Space",
                "genre_Sports",
                "genre_Super Power",
                "genre_Supernatural",
                "genre_Thriller",
                "genre_Vampire",
                "genre_Yaoi",
            ]
            + [f"user_anime_w_{i}" for i in range(128)]
            + [f"user_anime_h_{i}" for i in range(128)]
            + [f"user_genres_w_{i}" for i in range(128)]
            + [f"user_genres_h_{i}" for i in range(128)]
            + [f"user_producers_w_{i}" for i in range(128)]
            + [f"user_producers_h_{i}" for i in range(128)]
            + [f"user_licensors_w_{i}" for i in range(128)]
            + [f"user_licensors_h_{i}" for i in range(128)]
            + [f"user_studios_w_{i}" for i in range(128)]
            + [f"user_studios_h_{i}" for i in range(128)]
            + [f"user_source_w_{i}" for i in range(128)]
            + [f"user_source_h_{i}" for i in range(128)]
            + [f"user_ip_w_{i}" for i in range(128)]
            + [f"user_ip_h_{i}" for i in range(128)]
        )

    @property
    def categorical_features(self) -> list[str]:
        return []

    @property
    def continuous_features(self) -> list[str]:
        result = []
        for col in self.all_features:
            if col not in self.categorical_features:
                result.append(col)

        return result

    @property
    def targets(self) -> list[str]:
        return ["score"]

    def processing_train(self) -> None:
        self.train_X = self.train[self.all_features]
        self.train_y = self.train[self.targets]

    def processing_valid(self) -> None:
        self.valid_X = self.valid[self.all_features]
        self.valid_y = self.valid[self.targets]

    def processing_test(self) -> None:
        self.test_X = self.test[self.all_features]
