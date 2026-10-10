export SPARK_HOME=${SPARK_HOME:-/opt/spark}

SPARK_PACKAGES="org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.4,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0,org.apache.hadoop:hadoop-aws:3.3.4"

# Execute the Spark job
exec "${SPARK_HOME}/bin/spark-submit" \
    --packages "${SPARK_PACKAGES}" \
    --master local[*] \
    /app/spark/streaming_dual_sink.py
