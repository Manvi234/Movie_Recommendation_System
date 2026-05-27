"""
Evaluation metrics for the recommendation model.
Implements Hit Rate @ K and NDCG @ K using sampled-negative evaluation:
  For each test positive, rank it against 99 random negatives (100 items total).
  This is the standard NeuMF / BPR evaluation protocol for MovieLens.

Quality gate thresholds:
  HR@10  >= 0.15
  NDCG@10 >= 0.10
"""
import os
import pickle

import keras
import numpy as np
import pandas as pd
from models.two_tower import L2Normalize  # noqa: F401 — registers custom layer before load_model


MODEL_DIR = "data/models"
DATA_DIR = "data/processed"

K = 10
HR_THRESHOLD = 0.15
NDCG_THRESHOLD = 0.10
NUM_NEG_SAMPLES = 99  # rank target against 99 random negatives → 100 items total

USER_FEATURE_COLS = [
    "rating_count", "avg_rating", "rating_variance", "recency_score",
    "pref_action", "pref_adventure", "pref_animation", "pref_children",
    "pref_comedy", "pref_crime", "pref_documentary", "pref_drama",
    "pref_fantasy", "pref_film_noir", "pref_horror", "pref_musical",
    "pref_mystery", "pref_romance", "pref_sci_fi", "pref_thriller",
    "pref_war", "pref_western",
]
ITEM_FEATURE_COLS = [
    "avg_rating_received", "num_ratings", "decade",
    "genre_action", "genre_adventure", "genre_animation", "genre_children",
    "genre_comedy", "genre_crime", "genre_documentary", "genre_drama",
    "genre_fantasy", "genre_film_noir", "genre_horror", "genre_musical",
    "genre_mystery", "genre_romance", "genre_sci_fi", "genre_thriller",
    "genre_war", "genre_western",
]


def ndcg_at_k(hit_position: int | None, k: int) -> float:
    if hit_position is None or hit_position >= k:
        return 0.0
    return 1.0 / np.log2(hit_position + 2)


def load_delta_as_pandas(path: str) -> pd.DataFrame:
    parquet_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")
    ]
    return pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)


def evaluate(k: int = K) -> dict[str, float]:
    user_tower = keras.models.load_model(f"{MODEL_DIR}/user_tower.keras")
    item_embeddings = np.load(f"{MODEL_DIR}/item_embeddings.npy")  # (num_items, emb_dim)
    item_ids = np.load(f"{MODEL_DIR}/item_ids.npy")                # actual movieIds

    with open(f"{MODEL_DIR}/user_to_idx.pkl", "rb") as f:
        user_to_idx = pickle.load(f)
    with open(f"{MODEL_DIR}/user_scaler.pkl", "rb") as f:
        user_scaler = pickle.load(f)

    training_data = load_delta_as_pandas(f"{DATA_DIR}/training_data")
    user_features = load_delta_as_pandas(f"{DATA_DIR}/user_features")

    # Test set: time-based positive interactions only
    test_df = training_data[
        (training_data["split"] == "test") & (training_data["label"] == 1)
    ].reset_index(drop=True)
    test_df = test_df.merge(user_features, on="userId", how="left")
    for col in USER_FEATURE_COLS:
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(0.0)

    # All movieIds seen per user (to exclude from negatives)
    user_seen = training_data.groupby("userId")["movieId"].apply(set).to_dict()
    all_movie_ids = item_ids.tolist()
    movie_id_to_emb_idx = {int(mid): i for i, mid in enumerate(item_ids)}

    rng = np.random.default_rng(42)
    hits, ndcgs = [], []

    for user_id, group in test_df.groupby("userId"):
        if user_id not in user_to_idx:
            continue
        uid_idx = user_to_idx[user_id]

        raw_feats = group[USER_FEATURE_COLS].iloc[0].values.reshape(1, -1)
        u_feats = user_scaler.transform(raw_feats).astype(np.float32)
        u_vec = user_tower(
            {"user_id": np.array([[uid_idx]], dtype=np.int32), "user_features": u_feats},
            training=False,
        ).numpy().flatten()  # (emb_dim,)

        seen = user_seen.get(user_id, set())
        candidate_pool = [m for m in all_movie_ids if m not in seen]

        for _, row in group.iterrows():
            target_movie = int(row["movieId"])
            if target_movie not in movie_id_to_emb_idx:
                continue

            # Sample 99 negatives from unseen movies
            neg_sample = rng.choice(
                candidate_pool, size=min(NUM_NEG_SAMPLES, len(candidate_pool)), replace=False
            ).tolist()
            eval_movies = [target_movie] + neg_sample  # 100 items

            emb_indices = [movie_id_to_emb_idx[m] for m in eval_movies if m in movie_id_to_emb_idx]
            eval_embeddings = item_embeddings[emb_indices]  # (100, emb_dim)
            scores = eval_embeddings @ u_vec  # (100,)

            # Rank: position of target (index 0) in descending sorted scores
            rank = int(np.sum(scores > scores[0]))  # how many items scored higher

            hits.append(int(rank < k))
            ndcgs.append(ndcg_at_k(rank, k))

    hr = float(np.mean(hits))
    ndcg = float(np.mean(ndcgs))
    print(f"HR@{k}: {hr:.4f}  |  NDCG@{k}: {ndcg:.4f}  (sampled-negative protocol, {NUM_NEG_SAMPLES+1} items)")
    return {"hr": hr, "ndcg": ndcg}


def quality_gate(metrics: dict[str, float]) -> bool:
    passed = metrics["hr"] >= HR_THRESHOLD and metrics["ndcg"] >= NDCG_THRESHOLD
    status = "PASSED" if passed else "FAILED"
    print(
        f"Quality gate {status}: HR@{K}={metrics['hr']:.4f} (min {HR_THRESHOLD}), "
        f"NDCG@{K}={metrics['ndcg']:.4f} (min {NDCG_THRESHOLD})"
    )
    return passed


if __name__ == "__main__":
    metrics = evaluate()
    passed = quality_gate(metrics)
    exit(0 if passed else 1)
