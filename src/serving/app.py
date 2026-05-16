"""
FastAPI recommendation server.

Endpoints:
  GET  /recommend/{user_id}?n=10   → top-N movie recommendations
  GET  /health                      → model version and status
  POST /feedback                    → log user events back to Kafka
"""
import json
import os
import pickle
import time
from contextlib import asynccontextmanager
from typing import Any

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from models.two_tower import L2Normalize  # registers the custom layer before loading
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from confluent_kafka import Producer as KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False


MODEL_DIR = "data/models"
DATA_DIR = "data/processed"

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


class State:
    user_tower: Any = None
    item_embeddings: np.ndarray = None
    item_ids: np.ndarray = None
    user_to_idx: dict = None
    user_features: pd.DataFrame = None
    movies: pd.DataFrame = None
    kafka_producer: Any = None
    loaded_at: float = 0.0


state = State()


def load_delta_as_pandas(path: str) -> pd.DataFrame:
    parquet_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")
    ]
    return pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model artifacts...")
    state.user_tower = keras.models.load_model(f"{MODEL_DIR}/user_tower.keras")
    state.item_embeddings = np.load(f"{MODEL_DIR}/item_embeddings.npy")
    state.item_ids = np.load(f"{MODEL_DIR}/item_ids.npy")

    with open(f"{MODEL_DIR}/user_to_idx.pkl", "rb") as f:
        state.user_to_idx = pickle.load(f)

    state.user_features = load_delta_as_pandas(f"{DATA_DIR}/user_features")
    for col in USER_FEATURE_COLS:
        if col in state.user_features.columns:
            state.user_features[col] = state.user_features[col].fillna(0.0)

    # Load movie metadata for title lookup
    movies_path = "data/raw/ml-latest-small/movies.csv"
    if os.path.exists(movies_path):
        state.movies = pd.read_csv(movies_path)

    if KAFKA_AVAILABLE:
        try:
            state.kafka_producer = KafkaProducer(
                {"bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP", "localhost:29092")}
            )
        except Exception:
            pass  # serving works without Kafka

    state.loaded_at = time.time()
    print("Ready.")
    yield
    if state.kafka_producer:
        state.kafka_producer.flush()


app = FastAPI(title="Movie Recommender", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "loaded_at": state.loaded_at,
        "num_items": len(state.item_ids) if state.item_ids is not None else 0,
    }


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10):
    if user_id not in state.user_to_idx:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in training data")

    uid_idx = state.user_to_idx[user_id]
    user_row = state.user_features[state.user_features["userId"] == user_id]
    if user_row.empty:
        u_feats = np.zeros((1, len(USER_FEATURE_COLS)), dtype=np.float32)
    else:
        u_feats = user_row[USER_FEATURE_COLS].iloc[0].values.astype(np.float32).reshape(1, -1)

    u_vec = state.user_tower(
        {
            "user_id": np.array([[uid_idx]], dtype=np.int32),
            "user_features": u_feats,
        },
        training=False,
    ).numpy()

    # Dot product against pre-computed item matrix (fast)
    scores = state.item_embeddings @ u_vec.T
    top_idx = np.argsort(scores[:, 0])[::-1][:n]
    top_movie_ids = state.item_ids[top_idx].tolist()
    top_scores = scores[top_idx, 0].tolist()

    results = []
    for movie_id, score in zip(top_movie_ids, top_scores):
        entry: dict[str, Any] = {"movieId": int(movie_id), "score": round(float(score), 4)}
        if state.movies is not None:
            row = state.movies[state.movies["movieId"] == movie_id]
            if not row.empty:
                entry["title"] = row.iloc[0]["title"]
                entry["genres"] = row.iloc[0]["genres"]
        results.append(entry)

    return {"userId": user_id, "recommendations": results}


class NewUserRequest(BaseModel):
    genre_prefs: dict[str, float]   # e.g. {"horror": 1.0, "action": 0.0, ...}
    n: int = 10


@app.post("/recommend/new-user")
def recommend_new_user(req: NewUserRequest):
    """Content-based cold-start for users not in the training data."""
    liked_genres = {g for g, v in req.genre_prefs.items() if v > 0}

    # Build movieId → genre set lookup once
    genre_lookup: dict[int, set] = {}
    if state.movies is not None:
        for _, row in state.movies.iterrows():
            genre_lookup[int(row["movieId"])] = {
                g.lower().replace("-", "_") for g in str(row["genres"]).split("|")
            }

    item_embeddings = state.item_embeddings
    item_ids = state.item_ids

    matching_indices = [
        i for i, mid in enumerate(item_ids)
        if liked_genres & genre_lookup.get(int(mid), set())
    ]

    user_vec = (
        item_embeddings[matching_indices].mean(axis=0)
        if matching_indices
        else item_embeddings.mean(axis=0)
    )
    norm = np.linalg.norm(user_vec)
    if norm > 1e-8:
        user_vec = user_vec / norm

    scores = item_embeddings @ user_vec
    all_ranked = np.argsort(scores)[::-1]

    results = []
    for i in all_ranked:
        mid = int(item_ids[i])
        if not liked_genres or liked_genres & genre_lookup.get(mid, set()):
            entry: dict = {"movieId": mid, "score": round(float(scores[i]), 4)}
            if state.movies is not None:
                row = state.movies[state.movies["movieId"] == mid]
                if not row.empty:
                    entry["title"] = row.iloc[0]["title"]
                    entry["genres"] = row.iloc[0]["genres"]
            results.append(entry)
        if len(results) >= req.n:
            break

    return {"recommendations": results}


class FeedbackEvent(BaseModel):
    userId: int
    movieId: int
    event_type: str  # "click", "skip", "rating"
    rating: float = 0.0


@app.post("/feedback")
def feedback(event: FeedbackEvent):
    payload = {
        "userId": event.userId,
        "movieId": event.movieId,
        "rating": event.rating,
        "timestamp": int(time.time()),
        "event_type": event.event_type,
    }
    if state.kafka_producer:
        state.kafka_producer.produce(
            "movie-events",
            key=str(event.userId),
            value=json.dumps(payload).encode("utf-8"),
        )
        state.kafka_producer.poll(0)
    return {"status": "ok", "event": payload}
