from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class KafkaConfig:
    BOOTSTRAP_SERVERS: ClassVar[list[str]] = ["localhost:19092"]

    TOPIC_NAME: str = 'heart-disease-stream'
    NUM_PARTITIONS: int = 1
    REPLICATION_FACTOR: int = 1

    PRODUCER_CONFIG: ClassVar[dict[str, Any]] = {
        'bootstrap_servers': BOOTSTRAP_SERVERS,
        'value_serializer': lambda v: v.encode('utf-8'),
        'key_serializer': lambda k: k.encode('utf-8'),
        'acks': 'all',
        'retries': 3,
        'max_in_flight_requests_per_connection': 1,
        'compression_type': 'gzip',
        'api_version': (2, 8, 0),
    }