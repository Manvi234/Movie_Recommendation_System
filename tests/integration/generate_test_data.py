"""
Generates synthetic MovieLens-shaped CSV files for CI integration tests.
Run before spark/integration tests to populate data/raw/ with minimal data.
"""
import os
import random

import pandas as pd

OUT_DIR = "data/raw/ml-latest-small"

# Remove if it exists as a symlink or file (not a real directory)
if os.path.islink(OUT_DIR) or (os.path.exists(OUT_DIR) and not os.path.isdir(OUT_DIR)):
    os.remove(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

NUM_USERS = 50
NUM_MOVIES = 200
NUM_RATINGS = 2000
GENRES = ["Action|Drama", "Comedy", "Thriller", "Romance|Drama", "Sci-Fi"]

random.seed(42)

ratings_rows = []
for _ in range(NUM_RATINGS):
    ratings_rows.append(
        {
            "userId": random.randint(1, NUM_USERS),
            "movieId": random.randint(1, NUM_MOVIES),
            "rating": random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
            "timestamp": random.randint(800_000_000, 1_600_000_000),
        }
    )
pd.DataFrame(ratings_rows).to_csv(f"{OUT_DIR}/ratings.csv", index=False)

movies_rows = [
    {
        "movieId": i,
        "title": f"Movie {i} ({1990 + i % 30})",
        "genres": GENRES[i % len(GENRES)],
    }
    for i in range(1, NUM_MOVIES + 1)
]
pd.DataFrame(movies_rows).to_csv(f"{OUT_DIR}/movies.csv", index=False)

tags_rows = [
    {"userId": random.randint(1, NUM_USERS), "movieId": random.randint(1, NUM_MOVIES),
     "tag": f"tag{random.randint(1,20)}", "timestamp": random.randint(800_000_000, 1_600_000_000)}
    for _ in range(200)
]
pd.DataFrame(tags_rows).to_csv(f"{OUT_DIR}/tags.csv", index=False)

print(f"Generated test data in {OUT_DIR}/")
