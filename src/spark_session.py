from pyspark.sql import SparkSession


def get_spark():
    spark = (
        SparkSession.builder
        .appName("StockPredictionBigData")
        .master("local[*]")
        .getOrCreate()
    )
    return spark