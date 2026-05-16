"""
Replays MovieLens ratings into the 'movie-events' Kafka topic in timestamp order.
Simulates a real-time event stream from historical data.
"""
import argparse
import json
import time

import pandas as pd
from confluent_kafka import Producer


TOPIC = "movie-events"
BOOTSTRAP_SERVERS = "localhost:29092"


def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed for record {msg.key()}: {err}")


def run(ratings_path: str, limit: int | None, delay: float):
    df = pd.read_csv(ratings_path).sort_values("timestamp")
    if limit:
        df = df.head(limit)

    producer = Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})
    print(f"Streaming {len(df)} events to topic '{TOPIC}'...")

    for _, row in df.iterrows():
        event = {
            "userId": int(row["userId"]),
            "movieId": int(row["movieId"]),
            "rating": float(row["rating"]),
            "timestamp": int(row["timestamp"]),
            "event_type": "rating",
        }
        producer.produce(
            TOPIC,
            key=str(event["userId"]),
            value=json.dumps(event).encode("utf-8"),
            on_delivery=delivery_report,
        )
        producer.poll(0)
        if delay > 0:
            time.sleep(delay)

    producer.flush()
    print("Done streaming events.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default="data/raw/ml-latest-small/ratings.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.01)
    args = parser.parse_args()
    run(args.ratings, args.limit, args.delay)
