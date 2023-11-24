import pandas as pd
from ocha.preprocess.preprocess import Preprocess
from pandas import DataFrame
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder


class Atmacup15Preprocess(Preprocess):
    source: pd.DataFrame

    def preprocess(self) -> None:
        source = self.ip(self.source)
        source = self.user_scored_count(source)
        source = self.aired_year(source)
        source = self.duration_min(source)
        source = self.rates(source)
        source = self.user_embedding(source)

        self.source_processed = source

    def ip(self, source: DataFrame) -> DataFrame:
        source["ip"] = (
            source["japanese_name"]
            .fillna("")
            .str.replace("劇場版", "", regex=False)
            .str.strip()
            .str.lower()
            .str.normalize("NFKC")
            .str.extract(r"^([^\s]+)")[0]
            .str.slice(0, 4)
        )

        return source

    def user_scored_count(self, source: DataFrame) -> DataFrame:
        user_count = source.groupby("user_id").size().rename("user_count").to_frame()

        user_genre_count = (
            source.assign(genres=lambda df: df["genres"].str.split(", "))
            .explode("genres")
            .groupby(["user_id", "genres"])
            .size()
            .rename("count")
            .reset_index()
            .pivot(index="user_id", columns="genres", values="count")
            .fillna(0.0)
        )
        user_genre_count.columns = "genre_" + user_genre_count.columns

        user_ip_count = source.groupby(["user_id", "ip"]).size().rename("user_ip_count").to_frame()

        source = (
            source.merge(user_count, on="user_id", how="left")
            .merge(user_genre_count, on="user_id", how="left")
            .merge(user_ip_count, on=["user_id", "ip"], how="left")
        )

        return source

    def aired_year(self, source: DataFrame) -> DataFrame:
        source["aired_year"] = source["aired"].str.extract(r"(\d{4})").astype(float)

        return source

    def duration_min(self, source: DataFrame) -> DataFrame:
        source["duration_min"] = source["duration"].str.extract(r"(\d+) hr.").astype(float).fillna(0.0) * 60 + source[
            "duration"
        ].str.extract(r"(\d+) min.").astype(float).fillna(0.0)

        return source

    def rates(self, source: DataFrame) -> DataFrame:
        source["dropped_rate"] = source["dropped"] / source["plan_to_watch"]
        source["completed_rate"] = source["completed"] / source["plan_to_watch"]
        source["watching_rate"] = source["watching"] / source["plan_to_watch"]

        return source

    def user_embedding(
        self,
        source: DataFrame,
        n_components: int = 128,
        alpha_W: float = 0.01,
        max_iter: int = 1000,
        random_state: int = 1013,
    ) -> DataFrame:
        # Prepare encoders
        user_id_encoder = LabelEncoder()
        source["user_id_encoded"] = user_id_encoder.fit_transform(source["user_id"])

        anime_id_encoder = LabelEncoder()
        source["anime_id_encoded"] = anime_id_encoder.fit_transform(source["anime_id"])

        genres_encoder = LabelEncoder()
        source["genres_encoded"] = genres_encoder.fit_transform(source["genres"])

        producers_encoder = LabelEncoder()
        source["producers_encoded"] = producers_encoder.fit_transform(source["producers"])

        licensors_encoder = LabelEncoder()
        source["licensors_encoded"] = licensors_encoder.fit_transform(source["licensors"])

        studios_encoder = LabelEncoder()
        source["studios_encoded"] = studios_encoder.fit_transform(source["studios"])

        source_encoder = LabelEncoder()
        source["source_encoded"] = source_encoder.fit_transform(source["source"])

        ip_encoder = LabelEncoder()
        source["ip_encoded"] = ip_encoder.fit_transform(source["ip"])

        # user x anime
        user_anime_nmf = NMF(n_components=n_components, random_state=random_state, alpha_W=alpha_W, max_iter=max_iter)
        user_anime_count = pd.crosstab(source["user_id_encoded"], source["anime_id_encoded"])
        user_anime_w = user_anime_nmf.fit_transform(user_anime_count)
        user_anime_h = user_anime_nmf.components_

        user_anime_w_cols = [f"user_anime_w_{i}" for i in range(n_components)]
        user_anime_h_cols = [f"user_anime_h_{i}" for i in range(n_components)]

        user_anime_w = pd.DataFrame(user_anime_w, index=user_anime_count.index, columns=user_anime_w_cols)
        user_anime_h = pd.DataFrame(user_anime_h.T, index=user_anime_count.columns, columns=user_anime_h_cols)

        source = source.merge(user_anime_w, left_on="user_id_encoded", right_index=True, how="left").merge(
            user_anime_h, left_on="anime_id_encoded", right_index=True, how="left"
        )

        # user x genre
        user_genres_nmf = NMF(n_components=n_components, random_state=random_state, alpha_W=alpha_W, max_iter=max_iter)
        user_genres_count = pd.crosstab(source["user_id_encoded"], source["genres_encoded"])
        user_genres_w = user_genres_nmf.fit_transform(user_genres_count)
        user_genres_h = user_genres_nmf.components_

        user_genres_w_cols = [f"user_genres_w_{i}" for i in range(n_components)]
        user_genres_h_cols = [f"user_genres_h_{i}" for i in range(n_components)]

        user_genres_w = pd.DataFrame(user_genres_w, index=user_genres_count.index, columns=user_genres_w_cols)
        user_genres_h = pd.DataFrame(user_genres_h.T, index=user_genres_count.columns, columns=user_genres_h_cols)

        source = source.merge(user_genres_w, left_on="user_id_encoded", right_index=True, how="left").merge(
            user_genres_h, left_on="genres_encoded", right_index=True, how="left"
        )

        # user x producers
        user_producers_nmf = NMF(
            n_components=n_components, random_state=random_state, alpha_W=alpha_W, max_iter=max_iter
        )
        user_producers_count = pd.crosstab(source["user_id_encoded"], source["producers_encoded"])
        user_producers_w = user_producers_nmf.fit_transform(user_producers_count)
        user_producers_h = user_producers_nmf.components_

        user_producers_w_cols = [f"user_producers_w_{i}" for i in range(n_components)]
        user_producers_h_cols = [f"user_producers_h_{i}" for i in range(n_components)]

        user_producers_w = pd.DataFrame(
            user_producers_w, index=user_producers_count.index, columns=user_producers_w_cols
        )
        user_producers_h = pd.DataFrame(
            user_producers_h.T, index=user_producers_count.columns, columns=user_producers_h_cols
        )

        source = source.merge(user_producers_w, left_on="user_id_encoded", right_index=True, how="left").merge(
            user_producers_h, left_on="producers_encoded", right_index=True, how="left"
        )

        # user x licensors
        user_licensors_nmf = NMF(
            n_components=n_components, random_state=random_state, alpha_W=alpha_W, max_iter=max_iter
        )
        user_licensors_count = pd.crosstab(source["user_id_encoded"], source["licensors_encoded"])
        user_licensors_w = user_licensors_nmf.fit_transform(user_licensors_count)
        user_licensors_h = user_licensors_nmf.components_

        user_licensors_w_cols = [f"user_licensors_w_{i}" for i in range(n_components)]
        user_licensors_h_cols = [f"user_licensors_h_{i}" for i in range(n_components)]

        user_licensors_w = pd.DataFrame(
            user_licensors_w, index=user_licensors_count.index, columns=user_licensors_w_cols
        )
        user_licensors_h = pd.DataFrame(
            user_licensors_h.T, index=user_licensors_count.columns, columns=user_licensors_h_cols
        )

        source = source.merge(user_licensors_w, left_on="user_id_encoded", right_index=True, how="left").merge(
            user_licensors_h, left_on="licensors_encoded", right_index=True, how="left"
        )

        # user x studios
        user_studios_nmf = NMF(n_components=n_components, random_state=random_state, alpha_W=alpha_W, max_iter=max_iter)
        user_studios_count = pd.crosstab(source["user_id_encoded"], source["studios_encoded"])
        user_studios_w = user_studios_nmf.fit_transform(user_studios_count)
        user_studios_h = user_studios_nmf.components_

        user_studios_w_cols = [f"user_studios_w_{i}" for i in range(n_components)]
        user_studios_h_cols = [f"user_studios_h_{i}" for i in range(n_components)]

        user_studios_w = pd.DataFrame(user_studios_w, index=user_studios_count.index, columns=user_studios_w_cols)
        user_studios_h = pd.DataFrame(user_studios_h.T, index=user_studios_count.columns, columns=user_studios_h_cols)

        source = source.merge(user_studios_w, left_on="user_id_encoded", right_index=True, how="left").merge(
            user_studios_h, left_on="studios_encoded", right_index=True, how="left"
        )

        # user x score
        user_source_nmf = NMF(n_components=n_components, random_state=random_state, alpha_W=alpha_W, max_iter=max_iter)
        user_source_count = pd.crosstab(source["user_id_encoded"], source["source_encoded"])
        user_source_w = user_source_nmf.fit_transform(user_source_count)
        user_source_h = user_source_nmf.components_

        user_source_w_cols = [f"user_source_w_{i}" for i in range(n_components)]
        user_source_h_cols = [f"user_source_h_{i}" for i in range(n_components)]

        user_source_w = pd.DataFrame(user_source_w, index=user_source_count.index, columns=user_source_w_cols)
        user_source_h = pd.DataFrame(user_source_h.T, index=user_source_count.columns, columns=user_source_h_cols)

        source = source.merge(user_source_w, left_on="user_id_encoded", right_index=True, how="left").merge(
            user_source_h, left_on="source_encoded", right_index=True, how="left"
        )

        # user x ip
        user_ip_nmf = NMF(n_components=n_components, random_state=random_state, alpha_W=alpha_W, max_iter=max_iter)
        user_ip_count = pd.crosstab(source["user_id_encoded"], source["ip_encoded"])
        user_ip_w = user_ip_nmf.fit_transform(user_ip_count)
        user_ip_h = user_ip_nmf.components_

        user_ip_w_cols = [f"user_ip_w_{i}" for i in range(n_components)]
        user_ip_h_cols = [f"user_ip_h_{i}" for i in range(n_components)]

        user_ip_w = pd.DataFrame(user_ip_w, index=user_ip_count.index, columns=user_ip_w_cols)
        user_ip_h = pd.DataFrame(user_ip_h.T, index=user_ip_count.columns, columns=user_ip_h_cols)

        source = source.merge(user_ip_w, left_on="user_id_encoded", right_index=True, how="left").merge(
            user_ip_h, left_on="ip_encoded", right_index=True, how="left"
        )

        self.preprocessing_objects = {
            "user_id_encoder": user_id_encoder,
            "anime_id_encoder": anime_id_encoder,
            "genres_encoder": genres_encoder,
            "producers_encoder": producers_encoder,
            "licensors_encoder": licensors_encoder,
            "studios_encoder": studios_encoder,
            "source_encoder": source_encoder,
            "ip_encoder": ip_encoder,
        }

        return source
