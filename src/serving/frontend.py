import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..","src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import keras
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"
MODEL_DIR = "data/models"
DATA_DIR = "data/processed"
PROFILES_FILE = "data/custom_users.json"

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

RATING_STYLES = {
    "Generous (avg 4.5★)": 4.5,
    "Average (avg 3.5★)": 3.5,
    "Strict (avg 2.5★)": 2.5,
}

st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommender")
st.markdown("Powered by a Two-Tower deep learning model trained on MovieLens.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_profiles() -> dict:
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE) as f:
            return json.load(f)
    return {}


def save_profiles(profiles: dict):
    os.makedirs(os.path.dirname(PROFILES_FILE), exist_ok=True)
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)


@st.cache_data
def load_user_info():
    parquet_files = [
        os.path.join(DATA_DIR, "user_features", f)
        for f in os.listdir(f"{DATA_DIR}/user_features") if f.endswith(".parquet")
    ]
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    df = df.drop_duplicates("userId")
    return df.sort_values("userId")[["userId", "rating_count", "avg_rating"]].reset_index(drop=True)


@st.cache_resource
def load_model_artifacts():
    from models.two_tower import L2Normalize  # noqa: F401
    user_tower = keras.models.load_model(f"{MODEL_DIR}/user_tower.keras")
    item_embeddings = np.load(f"{MODEL_DIR}/item_embeddings.npy")
    item_ids = np.load(f"{MODEL_DIR}/item_ids.npy")
    with open(f"{MODEL_DIR}/user_to_idx.pkl", "rb") as f:
        user_to_idx = pickle.load(f)
    with open(f"{MODEL_DIR}/user_scaler.pkl", "rb") as f:
        user_scaler = pickle.load(f)
    movies = pd.read_csv("data/raw/ml-latest-small/movies.csv")
    return user_tower, item_embeddings, item_ids, user_to_idx, user_scaler, movies


def build_user_feature_vector(profile: dict, user_scaler) -> np.ndarray:
    """Convert a saved profile dict into a scaled feature vector for the user tower."""
    avg_r = profile["avg_rating"]
    genre_prefs = profile["genre_prefs"]  # dict genre→0/1

    raw = [
        profile.get("rating_count", 50),
        avg_r,
        profile.get("rating_variance", 0.8),
        1.0,  # recency_score — new user is active now
    ] + [genre_prefs.get(g.lower().replace("-", "_"), 0) for g in ALL_GENRES]

    raw_arr = np.array(raw, dtype=np.float32).reshape(1, -1)
    return user_scaler.transform(raw_arr).astype(np.float32)


def get_recommendations_for_new_user(genre_prefs: dict, item_embeddings, item_ids, movies_df, n):
    """
    Content-based cold-start for new users.
    1. Build a genre-lookup so we avoid repeated DataFrame scans.
    2. Average the L2-normalized embeddings of genre-matching movies and
       re-normalize the result (averaging many unit vectors can shrink the
       magnitude toward zero, making scores random).
    3. Score ALL items, then keep only movies that share at least one
       preferred genre — guaranteeing genre-appropriate results.
    """
    liked_genres = {g for g, v in genre_prefs.items() if v > 0}

    # Pre-build movieId → genre set lookup
    genre_lookup: dict[int, set] = {}
    for _, row in movies_df.iterrows():
        genre_lookup[int(row["movieId"])] = {
            g.lower().replace("-", "_") for g in str(row["genres"]).split("|")
        }

    matching_indices = [
        i for i, mid in enumerate(item_ids)
        if liked_genres & genre_lookup.get(int(mid), set())
    ]

    if matching_indices:
        user_vec = item_embeddings[matching_indices].mean(axis=0)
    else:
        user_vec = item_embeddings.mean(axis=0)

    # Normalize so dot-product scores are meaningful
    norm = np.linalg.norm(user_vec)
    if norm > 1e-8:
        user_vec = user_vec / norm

    scores = item_embeddings @ user_vec

    # Rank by score, but only return movies that match preferred genres
    all_ranked = np.argsort(scores)[::-1]
    results = []
    for i in all_ranked:
        mid = int(item_ids[i])
        if not liked_genres or liked_genres & genre_lookup.get(mid, set()):
            results.append((mid, float(scores[i])))
        if len(results) >= n:
            break
    return results


def show_recommendations(recs, movies_df):
    for i, (movie_id, score) in enumerate(recs, 1):
        row = movies_df[movies_df["movieId"] == movie_id]
        title = row.iloc[0]["title"] if not row.empty else f"Movie {movie_id}"
        genres = row.iloc[0]["genres"] if not row.empty else ""
        with st.container(border=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"**{i}. {title}**")
                if genres:
                    tags = "  ".join(f"`{g}`" for g in genres.split("|"))
                    st.markdown(tags)
            with col2:
                st.metric("Score", f"{int(score * 100)}%")


# ── Load artifacts ────────────────────────────────────────────────────────────

try:
    user_info = load_user_info()
    user_tower, item_embeddings, item_ids, user_to_idx, user_scaler, movies_df = load_model_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Options")
    mode = st.radio("Mode", ["Existing User", "Add New User", "My Saved Profile"])
    n_recs = st.slider("Recommendations", min_value=5, max_value=20, value=10)

# ── Mode: Existing User ───────────────────────────────────────────────────────

if mode == "Existing User":
    st.subheader("Select a User")
    user_labels = [
        f"User {row.userId}  —  {int(row.rating_count)} ratings  (avg {row.avg_rating:.1f}★)"
        for row in user_info.itertuples()
    ]
    selected = st.selectbox("Choose a user", options=range(len(user_labels)),
                            format_func=lambda i: user_labels[i])
    user_id = int(user_info.iloc[selected]["userId"])

    if st.button("Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Fetching recommendations..."):
            try:
                resp = requests.get(f"{API_URL}/recommend/{user_id}?n={n_recs}", timeout=10)
                resp.raise_for_status()
                data = resp.json()
                recs = [(r["movieId"], r["score"]) for r in data["recommendations"]]
                st.success(f"Top {len(recs)} recommendations for User {user_id}")
                show_recommendations(recs, movies_df)
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Run:\n`PYTHONPATH=src uvicorn serving.app:app --port 8000`")
            except Exception as e:
                st.error(f"Error: {e}")

# ── Mode: Add New User ────────────────────────────────────────────────────────

elif mode == "Add New User":
    st.subheader("Create Your Profile")

    name = st.text_input("Your name", placeholder="e.g. Manvi")

    st.markdown("**How do you usually rate movies?**")
    rating_style = st.select_slider(
        "Rating style", options=list(RATING_STYLES.keys()), value="Average (avg 3.5★)"
    )

    st.markdown("**How many movies do you watch?**")
    watch_freq = st.select_slider(
        "Watching frequency",
        options=["Casual (< 20)", "Regular (20–100)", "Enthusiast (100+)"],
        value="Regular (20–100)",
    )
    rating_count_map = {"Casual (< 20)": 10, "Regular (20–100)": 50, "Enthusiast (100+)": 150}

    st.markdown("**Select your favourite genres** (pick all that apply)")
    cols = st.columns(3)
    genre_prefs = {}
    for i, genre in enumerate(ALL_GENRES):
        key = genre.lower().replace("-", "_")
        genre_prefs[key] = 1.0 if cols[i % 3].checkbox(genre, key=f"genre_{key}") else 0.0

    st.divider()

    col1, col2 = st.columns(2)
    get_recs = col1.button("Get Recommendations", type="primary", use_container_width=True)
    save_btn = col2.button("Save Profile & Get Recommendations", use_container_width=True)

    if get_recs or save_btn:
        if not name.strip():
            st.warning("Please enter your name.")
        elif not any(genre_prefs.values()):
            st.warning("Please select at least one genre.")
        else:
            profile = {
                "name": name.strip(),
                "avg_rating": RATING_STYLES[rating_style],
                "rating_count": rating_count_map[watch_freq],
                "rating_variance": 0.8,
                "genre_prefs": genre_prefs,
            }

            if save_btn:
                profiles = load_profiles()
                profiles[name.strip()] = profile
                save_profiles(profiles)
                st.success(f"Profile saved for **{name}**! Find it under 'My Saved Profile'.")

            with st.spinner("Generating personalised recommendations..."):
                recs = get_recommendations_for_new_user(
                    profile["genre_prefs"], item_embeddings, item_ids, movies_df, n_recs
                )
                st.success(f"Top {len(recs)} recommendations for **{name}**")
                show_recommendations(recs, movies_df)

# ── Mode: My Saved Profile ────────────────────────────────────────────────────

elif mode == "My Saved Profile":
    st.subheader("My Saved Profile")
    profiles = load_profiles()

    if not profiles:
        st.info("No saved profiles yet. Go to **Add New User** to create one.")
    else:
        selected_name = st.selectbox("Select profile", list(profiles.keys()))
        profile = profiles[selected_name]

        with st.expander("Profile details"):
            st.write(f"**Avg rating:** {profile['avg_rating']}★")
            st.write(f"**Watch frequency:** {profile['rating_count']} movies")
            liked = [g.replace("_", " ").title()
                     for g, v in profile["genre_prefs"].items() if v > 0]
            st.write(f"**Favourite genres:** {', '.join(liked) if liked else 'None'}")

        col1, col2 = st.columns(2)
        get_btn = col1.button("Get Recommendations", type="primary", use_container_width=True)
        del_btn = col2.button("Delete Profile", use_container_width=True)

        if del_btn:
            del profiles[selected_name]
            save_profiles(profiles)
            st.success(f"Profile **{selected_name}** deleted.")
            st.rerun()

        if get_btn:
            with st.spinner("Generating personalised recommendations..."):
                recs = get_recommendations_for_new_user(
                    profile["genre_prefs"], item_embeddings, item_ids, movies_df, n_recs
                )
                st.success(f"Top {len(recs)} recommendations for **{selected_name}**")
                show_recommendations(recs, movies_df)
