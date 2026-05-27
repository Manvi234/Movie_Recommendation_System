"""
Training script for the Two-Tower recommendation model.
Reads features from Delta Lake (via pandas after Spark writes them),
trains on ml-latest-small data, saves the model and towers.

Usage:
  python src/models/train.py
"""
import os
import pickle

import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.two_tower import build_two_tower_model


DATA_DIR = "data/processed"
MODEL_DIR = "data/models"

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

BATCH_SIZE = 1024
EPOCHS = 50
EMBEDDING_DIM = 32


def load_delta_as_pandas(path: str) -> pd.DataFrame:
    parquet_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")
    ]
    return pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)


def fill_na_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.copy().assign(**{c: df[c].fillna(0.0) for c in cols if c in df.columns})


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading features...")
    training_data = load_delta_as_pandas(f"{DATA_DIR}/training_data")
    user_features = load_delta_as_pandas(f"{DATA_DIR}/user_features")
    item_features = load_delta_as_pandas(f"{DATA_DIR}/item_features")

    training_data = training_data.merge(user_features, on="userId", how="left")
    training_data = training_data.merge(item_features, on="movieId", how="left")
    training_data = fill_na_features(training_data, USER_FEATURE_COLS + ITEM_FEATURE_COLS)

    # Build contiguous integer IDs for embeddings
    all_users = sorted(training_data["userId"].unique())
    all_items = sorted(training_data["movieId"].unique())
    user_to_idx = {u: i for i, u in enumerate(all_users)}
    item_to_idx = {m: i for i, m in enumerate(all_items)}

    with open(f"{MODEL_DIR}/user_to_idx.pkl", "wb") as f:
        pickle.dump(user_to_idx, f)
    with open(f"{MODEL_DIR}/item_to_idx.pkl", "wb") as f:
        pickle.dump(item_to_idx, f)

    # Fit scalers on ALL data so inference uses the same scale
    user_scaler = StandardScaler()
    item_scaler = StandardScaler()
    user_scaler.fit(training_data[USER_FEATURE_COLS].values)
    item_scaler.fit(training_data[ITEM_FEATURE_COLS].values)

    with open(f"{MODEL_DIR}/user_scaler.pkl", "wb") as f:
        pickle.dump(user_scaler, f)
    with open(f"{MODEL_DIR}/item_scaler.pkl", "wb") as f:
        pickle.dump(item_scaler, f)

    # Random stratified split so validation has both positives and negatives
    all_labeled = training_data[training_data["label"].isin([0, 1])].reset_index(drop=True)
    train_df, val_df = train_test_split(
        all_labeled, test_size=0.15, random_state=42, stratify=all_labeled["label"]
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    def make_inputs(df):
        u_feats = user_scaler.transform(df[USER_FEATURE_COLS].values).astype(np.float32)
        i_feats = item_scaler.transform(df[ITEM_FEATURE_COLS].values).astype(np.float32)
        return {
            "user_id": df["userId"].map(user_to_idx).values.reshape(-1, 1),
            "user_features": u_feats,
            "item_id": df["movieId"].map(item_to_idx).fillna(0).astype(int).values.reshape(-1, 1),
            "item_features": i_feats,
        }

    X_train, y_train = make_inputs(train_df), train_df["label"].values.astype(np.float32)
    X_val, y_val = make_inputs(val_df), val_df["label"].values.astype(np.float32)

    model, user_tower, item_tower = build_two_tower_model(
        num_users=len(all_users),
        num_items=len(all_items),
        user_feature_dim=len(USER_FEATURE_COLS),
        item_feature_dim=len(ITEM_FEATURE_COLS),
        embedding_dim=EMBEDDING_DIM,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=8, restore_best_weights=True, mode="max"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", factor=0.5, patience=4, mode="max"
        ),
    ]

    print("Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print("Saving model...")
    model.save(f"{MODEL_DIR}/two_tower.keras")
    user_tower.save(f"{MODEL_DIR}/user_tower.keras")
    item_tower.save(f"{MODEL_DIR}/item_tower.keras")

    # Pre-compute all item embeddings for fast serving
    print("Pre-computing item embeddings...")
    item_feat_df = item_features.drop_duplicates("movieId").set_index("movieId").reindex(all_items).reset_index()
    item_feat_df = fill_na_features(item_feat_df, ITEM_FEATURE_COLS)
    item_feat_matrix = item_scaler.transform(
        item_feat_df[ITEM_FEATURE_COLS].values
    ).astype(np.float32)
    item_ids_seq = np.array(list(range(len(all_items)))).reshape(-1, 1)

    item_embeddings = item_tower.predict(
        {"item_id": item_ids_seq, "item_features": item_feat_matrix},
        batch_size=512,
    )
    np.save(f"{MODEL_DIR}/item_embeddings.npy", item_embeddings)
    np.save(f"{MODEL_DIR}/item_ids.npy", np.array(all_items))

    print(f"Training complete. Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
