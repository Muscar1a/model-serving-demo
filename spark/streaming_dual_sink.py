import os

from pyspark.sql import SparkSession

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datetime import datetime, timezone

from pyspark.sql.functions import (
    col,
    current_timestamp,
    from_json,
    hour,
    to_date,
    to_timestamp,
)
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


class DualSinkStreaming:
    def __init__(self):
        # Kafka config
        self.kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        self.kafka_topic = os.getenv("KAFKA_TOPIC", os.getenv("KAKFA_TOPIC"))

        # Elasticsearch config
        self.es_nodes = os.getenv("ES_NODES")
        self.es_port = os.getenv("ES_PORT")
        self.es_index = os.getenv("ES_INDEX")

        # SeaweedFS (S3-compatible) config
        self.s3_endpoint = os.getenv(
            "S3_ENDPOINT",
            os.getenv("SEAWEEDFS_ENDPOINT", "http://stream-project-seaweedfs:8333"),
        )
        self.s3_access_key = os.getenv(
            "S3_ACCESS_KEY",
            os.getenv("SEAWEEDFS_ACCESS_KEY", "seaweedadmin"),
        )
        self.s3_secret_key = os.getenv(
            "S3_SECRET_KEY",
            os.getenv("SEAWEEDFS_SECRET_KEY", "seaweedadmin"),
        )
        self.s3_bucket = os.getenv(
            "S3_BUCKET",
            os.getenv("SEAWEEDFS_BUCKET", "ml-data"),
        )

        self.checkpoint_location = "/app/spark/checkpoint"
        self.spark = None

    def create_spark_session(self):
        """CreateSparksessionwithallrequiredconfigurations"""
        print("\n" + "=" * 70)
        print("INITIALIZINGSPARKSESSION(DUALSINK:ES+MinIO)")
        print("=" * 70)

        print("\n[1]ConfiguringSparksession...")
        print(f"Kafka: {self.kafka_bootstrap}")
        print(f"Elasticsearch: {self.es_nodes}:{self.es_port}")
        print(f"MinIO: {self.s3_endpoint}")

        self.spark = (
            SparkSession.builder.appName("HeartDisease-DualSink-ES-MinIO")
            .config(
                "spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.4,"
                "org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0,"
                "org.apache.hadoop:hadoop-aws:3.3.4",
            )
            .config("spark.hadoop.fs.s3a.endpoint", self.s3_endpoint)
            .config("spark.hadoop.fs.s3a.access.key", self.s3_access_key)
            .config("spark.hadoop.fs.s3a.secret.key", self.s3_secret_key)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config(
                "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
            )
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            .config("spark.sql.streaming.checkpointLocation", self.checkpoint_location)
            .config("spark.es.nodes", self.es_nodes)
            .config("spark.es.port", self.es_port)
            .config("spark.es.nodes.wan.only", "true")
            .config("spark.es.index.auto.create", "true")
            .getOrCreate()
        )

        self.spark.sparkContext.setLogLevel("WARN")

        print(f"\nSpark version: {self.spark.version}")
        print("Sparksessioncreatedsuccessfully!")

    def define_schema(self):
        data_schema = StructType(
            [
                StructField("age", FloatType(), True),
                StructField("sex", FloatType(), True),
                StructField("cp", FloatType(), True),
                StructField("trestbps", FloatType(), True),
                StructField("chol", FloatType(), True),
                StructField("fbs", FloatType(), True),
                StructField("restecg", FloatType(), True),
                StructField("thalach", FloatType(), True),
                StructField("exang", FloatType(), True),
                StructField("oldpeak", FloatType(), True),
                StructField("slope", FloatType(), True),
                StructField("ca", FloatType(), True),
                StructField("thal", FloatType(), True),
                StructField("target", FloatType(), True),
            ]
        )

        message_schema = StructType(
            [
                StructField("batch_number", IntegerType(), True),
                StructField("record_id", LongType(), True),
                StructField("timestamp", StringType(), True),
                StructField("data", data_schema, True),
            ]
        )

        return message_schema

    def read_from_kafka(self):
        print("\n" + "=" * 70)
        print("READINGFROMKAFKA")
        print("=" * 70)

        print("\n[2]ConnectingtoKafka...")
        print(f"Bootstrapservers:{self.kafka_bootstrap}")
        print(f"Topic: {self.kafka_topic}")

        assert self.spark is not None
        df = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.kafka_bootstrap)
            .option("subscribe", self.kafka_topic)
            .option("startingOffsets", "latest")
            .option("failOnDataLoss", "false")
            .load()
        )

        print("Connected to Kafka successfully!")
        return df

    def process_data(self, df):
        print("\n" + "=" * 70)
        print("PROCESSING DATA")
        print("=" * 70)

        print("\n[3] Parsing JSON message...")

        message_schema = self.define_schema()
        # Parse JSON from Kafka
        parsed_df = (
            df.selectExpr("Cast(value AS STRING) as json_str")
            .select(from_json(col("json_str"), message_schema).alias("data"))
            .select("data.*")
        )

        # Flaaten and add metadata
        processed_df = (
            parsed_df.select(
                col("batch_number"),
                col("record_id"),
                to_timestamp(col("timestamp")).alias("event_timestamp"),
                col("data.age").cast("integer").alias("age"),
                col("data.sex").cast("integer").alias("sex"),
                col("data.cp").cast("integer").alias("cp"),
                col("data.trestbps").cast("integer").alias("trestbps"),
                col("data.chol").cast("integer").alias("chol"),
                col("data.fbs").cast("integer").alias("fbs"),
                col("data.restecg").cast("integer").alias("restecg"),
                col("data.thalach").cast("integer").alias("thalach"),
                col("data.exang").cast("integer").alias("exang"),
                col("data.oldpeak").alias("oldpeak"),
                col("data.slope").cast("integer").alias("slope"),
                col("data.ca").cast("integer").alias("ca"),
                col("data.thal").cast("integer").alias("thal"),
                col("data.target").cast("integer").alias("target"),
            )
            .withColumn("processed_at", current_timestamp())
            .withColumn("date", to_date(col("event_timestamp")))
            .withColumn("hour", hour(col("event_timestamp")))
        )

        print("Data processing pipeline configured!")
        print("\nOutput schema:")
        processed_df.printSchema()

        return processed_df

    def write_dual_sink(self, df):
        print("\n" + "=" * 70)
        print("CONFIGURING DUAL SINK (ES + Seaweedfs")
        print("=" * 70)

        def foreach_bacth_function(batch_df, batch_id):
            record_count = batch_df.count()

            if record_count == 0:
                print(f"\n[Batch {batch_id}] Empty batch, skipping...")
                return

            print(f"\n{'=' * 70}")
            print(f"PROCESSING BATCH {batch_id}")
            print(f"{'=' * 70}")
            print(f"Records: {record_count}")
            print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

            print("\nSample record:")
            batch_df.select("record_id", "age", "sex", "target").show(1, truncate=False)

            print(f"\n[Batch {batch_id}] Writing to Elasticsearch...")
            try:
                batch_df.write.format("org.elasticsearch.spark.sql").option(
                    "es.resource", self.es_index
                ).option("es.nodes", self.es_nodes).option(
                    "es.port", self.es_port
                ).option("es.nodes.wan.only", "true").mode("append").save()
                print(f"Elasticsearch: {record_count} records written")
            except Exception as e:  # noqa: BLE001
                print(f"Elasticsearch write failed: {e}")

            # Seaweedfs
            print(f"\n[Batch {batch_id}] Writing to Seaweedfs (Parquet)...")
            try:
                output_path = f"s3a://{self.s3_bucket}/raw/heart-disease"

                batch_df.repartition(1).write.format("parquet").partitionBy(
                    "date", "hour"
                ).mode("append").option("compression", "snappy").save(output_path)

                print(f"Seaweedfs:{record_count} recordswritten")
                print(f" Path:{output_path}/date=.../hour=.../")
                print(" Format: Parquet(Snappycompressed)")
            except Exception as e:  # noqa: BLE001
                print(f"MinIOwritefailed: {e}")
                import traceback

                traceback.print_exc()

            print(f"\n[Batch {batch_id}] Completedsuccessfully!")
            print("=" * 70)

        query = (
            df.writeStream.foreachBatch(foreach_bacth_function)
            .trigger(processingTime="5 seconds")
            .option("checkpointLocation", self.checkpoint_location)
            .start()
        )
        return query

    def run(self):
        try:
            self.create_spark_session()
            kafka_df = self.read_from_kafka()
            processed_df = self.process_data(kafka_df)
            query = self.write_dual_sink(processed_df)
            query.awaitTermination()
        except KeyboardInterrupt:
            print("\n\nStopping streaming...")
            if self.spark:
                self.spark.stop()
            print("Spark session stopped")
        except Exception as e:  # noqa: BLE001
            print(f"\n\nError: {e}")
            import traceback

            traceback.print_exc()
            if self.spark:
                self.spark.stop()


if __name__ == "__main__":
    app = DualSinkStreaming()
    app.run()
