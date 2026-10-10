import json
import os
import time
from datetime import datetime, timezone
from typing import Any

# Set environment variable before C/C++ runtime library initialization
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
from kafka import KafkaAdminClient, KafkaProducer
from kafka.admin import NewTopic
from kafka.errors import KafkaError, TopicAlreadyExistsError
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

from kafka_producer.config import KafkaConfig


class KafkaStreamProducer:
    def __init__(self, config_module: KafkaConfig | None = None) -> None:
        self.config: KafkaConfig = config_module or KafkaConfig()
        self.producer: KafkaProducer | None = None
        self.synthesizer: GaussianCopulaSynthesizer | None = None
        self.original_data: pd.DataFrame | None = None
        self.total_sent: int = 0

    def connect_kafka(self) -> bool:
        print("\n" + "=" * 70)
        print("KAFKA CONNECTION")
        print("=" * 70)

        try:
            print(f"\n[1] Connecting to Kafka at {self.config.BOOTSTRAP_SERVERS}...")
            self.producer = KafkaProducer(**self.config.PRODUCER_CONFIG)
            print("Successfully connected to Kafka broker")

            print(f"\n[2] Checking topic '{self.config.TOPIC_NAME}'...")
            try:
                admin_client = KafkaAdminClient(
                    bootstrap_servers=self.config.BOOTSTRAP_SERVERS,
                    client_id="producer_admin",
                    api_version=(2, 8, 0),
                )

                topic = NewTopic(
                    name=self.config.TOPIC_NAME,
                    num_partitions=self.config.NUM_PARTITIONS,
                    replication_factor=self.config.REPLICATION_FACTOR,
                )

                try:
                    admin_client.create_topics([topic], validate_only=False)
                    print(f"Created topic '{self.config.TOPIC_NAME}'")
                except TopicAlreadyExistsError:
                    print(f"Topic '{self.config.TOPIC_NAME}' already exists")
                except KafkaError as e:
                    print(f"Note: {e}")
                    print("Topic will be auto-created on first message")
                finally:
                    admin_client.close()

            except KafkaError as e:
                print(f"Could not create topic manually: {e}")
                print(
                    "Topic will be auto-created on first message (auto-create enabled)"
                )

            return True
        except KafkaError as e:
            print(f"Error connecting to Kafka: {e}")
            print("\nMake sure Kafka is running:")
            print("   docker-compose up -d kafka")
            return False

    def load_and_train_model(
        self, data_path: str, categorical_columns: list[str]
    ) -> None:
        print("\n" + "=" * 70)
        print("SDV MODEL TRAINING")
        print("=" * 70)

        print(f"\n[1] Loading dataset from: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.original_data = pd.read_csv(data_path)
        print(
            f"Loaded {len(self.original_data)} records with {len(self.original_data.columns)} columns"
        )

        print("\n[2] Creating metadata...")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(self.original_data)

        for col in categorical_columns:
            if col in self.original_data.columns:
                metadata.update_column(column_name=col, sdtype="categorical")

        print(f"Configured {len(categorical_columns)} categorical columns")

        # Train model
        print("\n[3] Training GaussianCopulaSynthesizer...")
        self.synthesizer = GaussianCopulaSynthesizer(
            metadata=metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
        )

        self.synthesizer.fit(self.original_data)
        print("Model training completed")

    def generate_batch(self, batch_size: int) -> pd.DataFrame:
        if self.synthesizer is None:
            raise RuntimeError(
                "Model is not trained. Call load_and_train_model() first."
            )
        synthetic_data: pd.DataFrame = self.synthesizer.sample(num_rows=batch_size)
        return synthetic_data

    def send_to_kafka(self, data: pd.DataFrame, batch_number: int) -> int:
        if self.producer is None:
            raise RuntimeError(
                "Kafka producer is not connected. Call connect_kafka() first."
            )

        records_sent = 0

        for _, row in data.iterrows():
            try:
                message: dict[str, Any] = {
                    "batch_number": batch_number,
                    "record_id": self.total_sent + records_sent,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": row.to_dict(),
                }

                message_json = json.dumps(message)

                future = self.producer.send(
                    self.config.TOPIC_NAME,
                    value=message_json,
                    key=str(message["record_id"]),
                )

                future.get(timeout=10)
                records_sent += 1

            except KafkaError as e:
                print(f"Kafka error sending record {records_sent}: {e}")
            except (TypeError, ValueError) as e:
                print(f"Serialization error on record {records_sent}: {e}")

        self.producer.flush()
        self.total_sent += records_sent

        return records_sent

    def stream_continuous(
        self, batch_size: int = 10, interval_seconds: float = 2.0
    ) -> None:
        print("\n" + "=" * 70)
        print("CONTINUOUS STREAMING TO KAFKA")
        print("=" * 70)
        print(f"\nTopic: {self.config.TOPIC_NAME}")
        print(f"Batch size: {batch_size} records")
        print(f"Interval: {interval_seconds} seconds")
        print("\nPress Ctrl+C to stop...")
        print("=" * 70)

        batch_number = 0

        try:
            while True:
                batch_number += 1
                start_time = time.time()

                data = self.generate_batch(batch_size)

                sent = self.send_to_kafka(data, batch_number)
                elapsed = time.time() - start_time

                print(f"\n[Batch {batch_number}] Sent {sent} records to Kafka")
                print(f"Total sent: {self.total_sent}")
                print(f"Time: {elapsed:.2f}s")
                print(
                    f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
                )

                if batch_number == 1:
                    print("\nSample record:")
                    print(data.iloc[0].to_dict())

                sleep_time = max(0.0, interval_seconds - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("STREAM STOPPED BY USER")
            print("=" * 70)
            print(f"Total batches: {batch_number}")
            print(f"Total records sent: {self.total_sent}")
            print(f"Topic: {self.config.TOPIC_NAME}")

    def close(self) -> None:
        if self.producer:
            self.producer.close()
            print("\nKafka producer closed")


def main() -> None:
    data_path = "raw-data/Heart_disease_cleveland_new.csv"
    categorical_columns = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
        "target",
    ]

    batch_size = 10
    interval_seconds = 2.0

    producer = KafkaStreamProducer()

    try:
        if not producer.connect_kafka():
            return

        producer.load_and_train_model(data_path, categorical_columns)

        # Start continuous streaming
        producer.stream_continuous(
            batch_size=batch_size,
            interval_seconds=interval_seconds,
        )

    except Exception as e:  # noqa: BLE001
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
    finally:
        producer.close()


if __name__ == "__main__":
    main()
