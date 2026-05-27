"""
Nightly Spark batch pipeline.
Reads raw MovieLens files → engineers features → writes Delta Lake tables.
Run with:
  spark-submit --packages io.delta:delta-spark_2.12:3.2.1 src/spark/batch_pipeline.py
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw/ml-latest-small"
PROCESSED_DIR = f"{DATA_DIR}/processed"

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def build_spark():
    return (
        SparkSession.builder.appName("MovieRecommender-Batch")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def extract_year(title_col):
    return F.regexp_extract(title_col, r"\((\d{4})\)$", 1).cast("int")


def build_user_features(ratings, movies):
    max_ts = ratings.agg(F.max("timestamp")).collect()[0][0]
    w = Window.partitionBy("userId")

    base = (
        ratings
        .withColumn("rating_count", F.count("*").over(w))
        .withColumn("avg_rating", F.avg("rating").over(w))
        .withColumn("rating_variance", F.variance("rating").over(w))
        .withColumn("recency_score", F.max("timestamp").over(w) / max_ts)
        .select("userId", "movieId", "rating", "rating_count", "avg_rating",
                "rating_variance", "recency_score")
        .distinct()
    )

    # Join with movie genres so we can compute per-user genre preferences
    movies_genres = movies.select("movieId", "genres")
    base = base.join(movies_genres, "movieId", "left")

    # For each genre: fraction of user's positive ratings (>=3.5) that include this genre
    genre_pref_cols = []
    for g in ALL_GENRES:
        col_name = f"pref_{g.lower().replace('-', '_')}"
        genre_pref_cols.append(
            (
                F.sum(
                    F.when(
                        (F.col("rating") >= 3.5) &
                        F.array_contains(F.split("genres", r"\|"), g),
                        1
                    ).otherwise(0)
                ).over(w) /
                F.greatest(F.sum(F.when(F.col("rating") >= 3.5, 1).otherwise(0)).over(w), F.lit(1))
            ).alias(col_name)
        )

    return (
        base.withColumn("_dummy", F.lit(1))
        .select(
            "userId", "rating_count", "avg_rating", "rating_variance", "recency_score",
            *genre_pref_cols
        )
        .distinct()
    )


def build_item_features(ratings, movies):
    genre_cols = [
        F.when(F.array_contains(F.split("genres", r"\|"), g), 1).otherwise(0).alias(f"genre_{g.lower().replace('-','_')}")
        for g in ALL_GENRES
    ]
    items = movies.withColumn("release_year", extract_year("title"))
    items = items.withColumn("decade", (F.col("release_year") / 10).cast("int") * 10)

    agg = ratings.groupBy("movieId").agg(
        F.avg("rating").alias("avg_rating_received"),
        F.count("*").alias("num_ratings"),
    )
    return items.join(agg, "movieId", "left").select(
        "movieId", "title", "genres", "release_year", "decade",
        "avg_rating_received", "num_ratings",
        *genre_cols,
    )


def build_training_data(ratings):
    # Time-based split: most recent 10% of interactions → test set
    ts_sorted = ratings.orderBy("timestamp")
    n = ts_sorted.count()
    cutoff_idx = int(n * 0.9)
    cutoff_ts = ts_sorted.limit(cutoff_idx).agg(F.max("timestamp")).collect()[0][0]

    positives = ratings.filter(F.col("rating") >= 3.5).withColumn("label", F.lit(1))

    # 4:1 negative sampling — random movies the user hasn't rated
    all_movies = ratings.select("movieId").distinct()
    user_seen = ratings.select("userId", "movieId")

    # cross join then anti-join removes seen movies
    users = positives.select("userId").distinct()
    candidates = users.crossJoin(all_movies)
    negatives = (
        candidates.join(user_seen, ["userId", "movieId"], "left_anti")
        .withColumn("rating", F.lit(0.0))
        .withColumn("label", F.lit(0))
        .withColumn("timestamp", F.lit(0))
    )

    # Sample 4 negatives per positive
    n_positives = positives.count()
    neg_fraction = min(1.0, (4 * n_positives) / negatives.count())
    negatives = negatives.sample(fraction=neg_fraction, seed=42)

    combined = positives.select("userId", "movieId", "rating", "timestamp", "label").union(
        negatives.select("userId", "movieId", "rating", "timestamp", "label")
    )

    return (
        combined.withColumn(
            "split",
            F.when(F.col("timestamp") > cutoff_ts, "test").otherwise("train"),
        ),
        cutoff_ts,
    )


def write_delta(df, path, mode="overwrite"):
    df.write.format("delta").mode(mode).option("overwriteSchema", "true").save(path)
    print(f"Written to {path} ({df.count()} rows)")


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("Reading raw data...")
    ratings = spark.read.csv(f"{RAW_DIR}/ratings.csv", header=True, inferSchema=True)
    movies = spark.read.csv(f"{RAW_DIR}/movies.csv", header=True, inferSchema=True)

    print("Building user features...")
    user_features = build_user_features(ratings, movies)
    write_delta(user_features, f"{PROCESSED_DIR}/user_features")

    print("Building item features...")
    item_features = build_item_features(ratings, movies)
    write_delta(item_features, f"{PROCESSED_DIR}/item_features")

    print("Building training data...")
    training_data, cutoff_ts = build_training_data(ratings)
    write_delta(training_data, f"{PROCESSED_DIR}/training_data")
    print(f"Train/test split at timestamp: {cutoff_ts}")

    print("Batch pipeline complete.")
    spark.stop()


if __name__ == "__main__":
    main()
