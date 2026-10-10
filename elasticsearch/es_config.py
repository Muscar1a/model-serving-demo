from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class ElasticsearchConfig:
    ES_HOST: ClassVar[tuple[str, ...]] = ("localhost",)
    ES_PORT: int = 19200
    ES_HTTP_AUTH: str | None = None

    INDEX_NAME = "heart-disease-stream"

    INDEX_SETTINGS: ClassVar[dict[str, Any]] = {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "1s",
    }

    INDEX_MAPPING: ClassVar[dict[str, Any]] = {
        "properties": {
            "batch_number": {"type": "integer"},
            "record_id": {"type": "long"},
            "timestamp": {
                "type": "date",
                "format": "strict_date_optional_time||epoch_millis",
            },
            "age": {"type": "integer"},
            "sex": {"type": "keyword"},
            "cp": {"type": "keyword"},
            "trestbps": {"type": "integer"},
            "chol": {"type": "integer"},
            "fbs": {"type": "keyword"},
            "restecg": {"type": "keyword"},
            "thalach": {"type": "integer"},
            "exang": {"type": "keyword"},
            "oldpeak": {"type": "float"},
            "slope": {"type": "keyword"},
            "ca": {"type": "keyword"},
            "target": {"type": "keyword"},
        }
    }
