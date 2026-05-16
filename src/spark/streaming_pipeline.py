"""
Spark Structured Streaming pipeline.
Consumes 'movie-events' from Kafka → writes to two Delta Lake tables:
  • data/processed/live_events   — every raw event as it arrives
  • data/processed/user_activity — per-user rolling aggregates (5-min micro-batch)

Run with:
  spark-submit \
    --packages io.delta:delta-spark_2.12:3.2.1,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 \
    src/spark/streaming_pipeline.py
"""
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, LongType, StringType, StructField, StructType,
)


KAFKA_BOOTSTRAP = "localhost:29092"
TOPIC = "movie-events"
CHECKPOINT_EVENTS = "data/checkpoints/live_events"
CHECKPOINT_ACTIVITY = "data/checkpoints/user_activity"
OUTPUT_EVENTS = "data/processed/live_events"
OUTPUT_ACTIVITY = "data/processed/user_activity"

EVENT_SCHEMA = StructType([
    StructField("userId", IntegerType()),
    StructField("movieId", IntegerType()),
    StructField("rating", DoubleType()),
    StructField("timestamp", LongType()),
    StructField("event_type", StringType()),
])


def build_spark():
    return (
        SparkSession.builder.appName("MovieRecommender-Streaming")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def parse_stream(spark):
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )
    return (
        raw.select(
            F.from_json(F.col("value").cast("string"), EVENT_SCHEMA).alias("d")
        )
        .select("d.*")
        .withColumn("event_time", F.to_timestamp(F.col("timestamp").cast("double")))
    )


def write_raw_events(parsed):
    """Append every raw event to Delta as it arrives."""
    def write_batch(batch_df, _):
        if batch_df.count() == 0:
            return
        (
            batch_df.write.format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .save(OUTPUT_EVENTS)
        )

    return (
        parsed.writeStream
        .foreachBatch(write_batch)
        .option("checkpointLocation", CHECKPOINT_EVENTS)
        .trigger(processingTime="10 seconds")
        .start()
    )


def write_user_activity(parsed):
    """
    Rolling per-user aggregates over a 1-hour window, sliding every 5 minutes.
    Uses foreachBatch + Delta MERGE so the activity table is always up to date.
    """
    from delta.tables import DeltaTable

    def upsert_activity(batch_df, _):
        if batch_df.count() == 0:
            return

        agg = (
            batch_df
            .withWatermark("event_time", "10 minutes")
            .groupBy(
                F.window("event_time", "1 hour", "5 minutes").alias("window"),
                "userId",
            )
            .agg(
                F.count("*").alias("event_count"),
                F.avg("rating").alias("avg_rating_in_window"),
                F.max("event_time").alias("last_event_time"),
            )
            .withColumn("window_start", F.col("window.start"))
            .withColumn("window_end", F.col("window.end"))
            .drop("window")
        )

        if not os.path.exists(OUTPUT_ACTIVITY):
            agg.write.format("delta").mode("overwrite").save(OUTPUT_ACTIVITY)
            return

        target = DeltaTable.forPath(batch_df.sparkSession, OUTPUT_ACTIVITY)
        (
            target.alias("t")
            .merge(
                agg.alias("s"),
                "t.userId = s.userId AND t.window_start = s.window_start",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    return (
        parsed.writeStream
        .foreachBatch(upsert_activity)
        .option("checkpointLocation", CHECKPOINT_ACTIVITY)
        .trigger(processingTime="30 seconds")
        .start()
    )


def main():
    for path in [CHECKPOINT_EVENTS, CHECKPOINT_ACTIVITY]:
        os.makedirs(path, exist_ok=True)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"Connecting to Kafka at {KAFKA_BOOTSTRAP}, topic='{TOPIC}'")
    parsed = parse_stream(spark)

    q_events = write_raw_events(parsed)
    q_activity = write_user_activity(parsed)

    print("Streaming started. Waiting for events...")
    print(f"  Raw events  → {OUTPUT_EVENTS}")
    print(f"  User activity → {OUTPUT_ACTIVITY}")

    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        print("Stopping streams...")
        q_events.stop()
        q_activity.stop()


if __name__ == "__main__":
    main()
