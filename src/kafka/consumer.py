"""
Test consumer — reads from 'movie-events' and prints messages.
This is for verification only; the real consumer is Spark Streaming.
"""
import json

from confluent_kafka import Consumer, KafkaError


TOPIC = "movie-events"
BOOTSTRAP_SERVERS = "localhost:29092"


def run(max_messages: int = 100):
    consumer = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP_SERVERS,
            "group.id": "test-consumer-group",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([TOPIC])
    print(f"Listening on '{TOPIC}'... (Ctrl-C to stop)")

    count = 0
    try:
        while count < max_messages:
            msg = consumer.poll(timeout=5.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise Exception(msg.error())
            event = json.loads(msg.value().decode("utf-8"))
            print(f"[offset={msg.offset()}] {event}")
            count += 1
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


if __name__ == "__main__":
    run()
