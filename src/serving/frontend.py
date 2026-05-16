import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import keras
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
MODEL_DIR = "data/models"
DATA_DIR = "data/processed"
PROFILES_FILE = "data/custom_users.json"

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

GENRE_EMOJI = {
    "Action": "💥", "Adventure": "🗺️", "Animation": "🎨", "Children": "🧒",
    "Comedy": "😂", "Crime": "🔍", "Documentary": "🎥", "Drama": "🎭",
    "Fantasy": "🧙", "Film-Noir": "🕵️", "Horror": "👻", "Musical": "🎵",
    "Mystery": "🔮", "Romance": "❤️", "Sci-Fi": "🚀", "Thriller": "😱",
    "War": "⚔️", "Western": "🤠",
}

RATING_STYLES = {
    "Generous (avg 4.5★)": 4.5,
    "Average (avg 3.5★)": 3.5,
    "Strict (avg 2.5★)": 2.5,
}

# ── Page config & custom CSS ──────────────────────────────────────────────────

st.set_page_config(page_title="🎬 CineMatch", page_icon="🎬", layout="wide")

st.markdown("""
<style>
/* Dark cinema background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid #e94560;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* Header */
.cinema-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.cinema-title {
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(90deg, #e94560, #f5a623, #e94560);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 2px;
}
.cinema-subtitle {
    color: #a0a0b0;
    font-size: 1rem;
    margin-top: -0.5rem;
}

/* Stat cards */
.stat-row { display: flex; gap: 1rem; margin: 1rem 0; }
.stat-card {
    flex: 1;
    background: rgba(233, 69, 96, 0.1);
    border: 1px solid rgba(233, 69, 96, 0.3);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-number { font-size: 1.8rem; font-weight: 700; color: #e94560; }
.stat-label { font-size: 0.8rem; color: #a0a0b0; text-transform: uppercase; letter-spacing: 1px; }

/* Movie cards */
.movie-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.movie-card:hover { border-color: rgba(233, 69, 96, 0.5); }
.movie-rank {
    font-size: 0.85rem;
    color: #e94560;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.movie-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0.2rem 0 0.5rem 0;
}
.genre-pill {
    display: inline-block;
    background: rgba(233, 69, 96, 0.15);
    border: 1px solid rgba(233, 69, 96, 0.3);
    color: #e94560;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px 3px 2px 0;
    font-weight: 500;
}
.score-bar-container {
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
    height: 6px;
    margin-top: 0.8rem;
    overflow: hidden;
}
.score-bar {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #e94560, #f5a623);
}
.score-label {
    font-size: 0.8rem;
    color: #a0a0b0;
    margin-top: 0.3rem;
}

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffffff;
    border-left: 4px solid #e94560;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

/* Profile card */
.profile-card {
    background: rgba(233, 69, 96, 0.08);
    border: 1px solid rgba(233, 69, 96, 0.25);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.profile-name { font-size: 1.3rem; font-weight: 700; color: #fff; }
.profile-detail { font-size: 0.9rem; color: #a0a0b0; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)


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
    avg_r = profile["avg_rating"]
    genre_prefs = profile["genre_prefs"]
    raw = [
        profile.get("rating_count", 50),
        avg_r,
        profile.get("rating_variance", 0.8),
        1.0,
    ] + [genre_prefs.get(g.lower().replace("-", "_"), 0) for g in ALL_GENRES]
    raw_arr = np.array(raw, dtype=np.float32).reshape(1, -1)
    return user_scaler.transform(raw_arr).astype(np.float32)


def get_recommendations_for_new_user(genre_prefs: dict, item_embeddings, item_ids, movies_df, n):
    liked_genres = {g for g, v in genre_prefs.items() if v > 0}
    genre_lookup: dict[int, set] = {}
    for _, row in movies_df.iterrows():
        genre_lookup[int(row["movieId"])] = {
            g.lower().replace("-", "_") for g in str(row["genres"]).split("|")
        }
    matching_indices = [
        i for i, mid in enumerate(item_ids)
        if liked_genres & genre_lookup.get(int(mid), set())
    ]
    user_vec = item_embeddings[matching_indices].mean(axis=0) if matching_indices else item_embeddings.mean(axis=0)
    norm = np.linalg.norm(user_vec)
    if norm > 1e-8:
        user_vec = user_vec / norm
    scores = item_embeddings @ user_vec
    all_ranked = np.argsort(scores)[::-1]
    results = []
    for i in all_ranked:
        mid = int(item_ids[i])
        if not liked_genres or liked_genres & genre_lookup.get(mid, set()):
            results.append((mid, float(scores[i])))
        if len(results) >= n:
            break
    return results


def score_color(score: float) -> str:
    if score >= 0.80:
        return "#2ecc71"
    elif score >= 0.65:
        return "#f5a623"
    else:
        return "#e94560"


def show_recommendations(recs, movies_df):
    for i, (movie_id, score) in enumerate(recs, 1):
        row = movies_df[movies_df["movieId"] == movie_id]
        title = row.iloc[0]["title"] if not row.empty else f"Movie {movie_id}"
        genres = row.iloc[0]["genres"] if not row.empty else ""

        genre_pills = ""
        if genres:
            for g in genres.split("|"):
                emoji = GENRE_EMOJI.get(g, "🎬")
                genre_pills += f'<span class="genre-pill">{emoji} {g}</span>'

        pct = int(score * 100)
        color = score_color(score)
        bar_width = pct

        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-rank">#{i}</div>
            <div class="movie-title">{title}</div>
            <div>{genre_pills}</div>
            <div class="score-bar-container">
                <div class="score-bar" style="width:{bar_width}%; background: linear-gradient(90deg, {color}, {color}cc);"></div>
            </div>
            <div class="score-label">Match score: <strong style="color:{color}">{pct}%</strong></div>
        </div>
        """, unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────

try:
    user_info = load_user_info()
    user_tower, item_embeddings, item_ids, user_to_idx, user_scaler, movies_df = load_model_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="cinema-header">
    <div class="cinema-title">🎬 CineMatch</div>
    <div class="cinema-subtitle">Personalised recommendations powered by a Two-Tower deep learning model</div>
</div>
""", unsafe_allow_html=True)

# Stats bar
total_users = len(user_info)
total_movies = len(movies_df)
total_profiles = len(load_profiles())
st.markdown(f"""
<div class="stat-row">
    <div class="stat-card">
        <div class="stat-number">{total_users:,}</div>
        <div class="stat-label">Users</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{total_movies:,}</div>
        <div class="stat-label">Movies</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">HR@10 = 0.77</div>
        <div class="stat-label">Model Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{total_profiles}</div>
        <div class="stat-label">Saved Profiles</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    mode = st.radio(
        "Mode",
        ["🎯 Existing User", "✨ New User", "👤 My Profile"],
        label_visibility="collapsed",
    )
    st.divider()
    n_recs = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)
    st.divider()
    st.markdown("<div style='color:#a0a0b0; font-size:0.8rem;'>Two-Tower Model · MovieLens · Keras</div>", unsafe_allow_html=True)

# ── Mode: Existing User ───────────────────────────────────────────────────────

if mode == "🎯 Existing User":
    st.markdown('<div class="section-header">Select a User</div>', unsafe_allow_html=True)

    user_labels = [
        f"User {row.userId}  —  {int(row.rating_count)} ratings  (avg {row.avg_rating:.1f}★)"
        for row in user_info.itertuples()
    ]
    selected = st.selectbox("Choose a user", options=range(len(user_labels)),
                            format_func=lambda i: user_labels[i], label_visibility="collapsed")
    user_id = int(user_info.iloc[selected]["userId"])

    sel_row = user_info.iloc[selected]
    st.markdown(f"""
    <div class="profile-card">
        <div class="profile-name">👤 User {int(sel_row.userId)}</div>
        <div class="profile-detail">⭐ {sel_row.avg_rating:.1f} avg rating &nbsp;|&nbsp; 🎬 {int(sel_row.rating_count)} movies rated</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("✨ Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Fetching recommendations..."):
            try:
                resp = requests.get(f"{API_URL}/recommend/{user_id}?n={n_recs}", timeout=10)
                resp.raise_for_status()
                data = resp.json()
                recs = [(r["movieId"], r["score"]) for r in data["recommendations"]]
                st.markdown(f'<div class="section-header">Top {len(recs)} picks for User {user_id}</div>', unsafe_allow_html=True)
                show_recommendations(recs, movies_df)
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Run:\n`PYTHONPATH=src uvicorn serving.app:app --port 8000`")
            except Exception as e:
                st.error(f"Error: {e}")

# ── Mode: Add New User ────────────────────────────────────────────────────────

elif mode == "✨ New User":
    st.markdown('<div class="section-header">Create Your Profile</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        name = st.text_input("Your name", placeholder="e.g. Manvi")
        rating_style = st.select_slider(
            "How do you usually rate movies?",
            options=list(RATING_STYLES.keys()),
            value="Average (avg 3.5★)",
        )
    with col_b:
        watch_freq = st.select_slider(
            "How many movies do you watch?",
            options=["Casual (< 20)", "Regular (20–100)", "Enthusiast (100+)"],
            value="Regular (20–100)",
        )

    rating_count_map = {"Casual (< 20)": 10, "Regular (20–100)": 50, "Enthusiast (100+)": 150}

    st.markdown('<div class="section-header">Favourite Genres</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    genre_prefs = {}
    for i, genre in enumerate(ALL_GENRES):
        key = genre.lower().replace("-", "_")
        genre_prefs[key] = 1.0 if cols[i % 6].checkbox(genre, key=f"genre_{key}") else 0.0

    st.divider()
    col1, col2 = st.columns(2)
    get_recs = col1.button("✨ Get Recommendations", type="primary", use_container_width=True)
    save_btn = col2.button("💾 Save Profile & Get Recommendations", use_container_width=True)

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
                st.success(f"Profile saved for **{name}**! Find it under 'My Profile'.")

            with st.spinner("Generating personalised recommendations..."):
                recs = get_recommendations_for_new_user(
                    profile["genre_prefs"], item_embeddings, item_ids, movies_df, n_recs
                )
                st.markdown(f'<div class="section-header">Top {len(recs)} picks for {name} 🎬</div>', unsafe_allow_html=True)
                show_recommendations(recs, movies_df)

# ── Mode: My Saved Profile ────────────────────────────────────────────────────

elif mode == "👤 My Profile":
    st.markdown('<div class="section-header">My Saved Profiles</div>', unsafe_allow_html=True)
    profiles = load_profiles()

    if not profiles:
        st.info("No saved profiles yet. Go to **✨ New User** to create one.")
    else:
        selected_name = st.selectbox("Select profile", list(profiles.keys()), label_visibility="collapsed")
        profile = profiles[selected_name]

        liked = [g.replace("_", " ").title() for g, v in profile["genre_prefs"].items() if v > 0]
        genre_pills = "".join(
            f'<span class="genre-pill">{GENRE_EMOJI.get(g.replace(" ", "-").title(), "🎬")} {g}</span>'
            for g in liked
        )
        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-name">👤 {selected_name}</div>
            <div class="profile-detail">⭐ {profile['avg_rating']} avg rating &nbsp;|&nbsp; 🎬 {profile['rating_count']} movies</div>
            <div style="margin-top:0.7rem">{genre_pills if genre_pills else '<span style="color:#a0a0b0">No genres selected</span>'}</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        get_btn = col1.button("✨ Get Recommendations", type="primary", use_container_width=True)
        del_btn = col2.button("🗑️ Delete Profile", use_container_width=True)

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
                st.markdown(f'<div class="section-header">Top {len(recs)} picks for {selected_name} 🎬</div>', unsafe_allow_html=True)
                show_recommendations(recs, movies_df)
