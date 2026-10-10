FROM spark:3.4.4-python3
USER root
RUN pip install --no-cache-dir pyspark==3.4.4 elasticsearch requests
WORKDIR /app
COPY spark/*.py /app/spark/
COPY spark/*.sh /app/spark/
RUN chmod +x /app/spark/*.sh
RUN mkdir -p /app/spark/checkpoint && chmod -R 777 /app/spark/checkpoint
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin
ENV PYTHONPATH=/app:$PYTHONPATH
CMD ["tail", "-f", "/dev/null"]