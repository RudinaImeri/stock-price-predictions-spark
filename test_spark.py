import os
from pyspark.sql import SparkSession

from src.ingestion.market_ingestion import load_stock_data
from src.preprocessing.feature_engineering_spark import add_features
from src.training.train_gbt import train_gbt
from src.prediction.predict import load_model, predict

python_path = r"C:\Users\Hp\anaconda3\envs\spark-env-1\python.exe"

os.environ["PYSPARK_PYTHON"] = python_path
os.environ["PYSPARK_DRIVER_PYTHON"] = python_path

spark = SparkSession.builder \
    .appName("StockPrediction") \
    .config("spark.pyspark.python", python_path) \
    .config("spark.pyspark.driver.python", python_path) \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 1. Load
pdf = load_stock_data("AAPL")

# 2. Convert
spark_df = spark.createDataFrame(pdf)
spark_df = spark_df.withColumnRenamed("Date", "date")

# 3. Features
featured_df = add_features(spark_df)

print("COUNT:", featured_df.count())

# 4. Train
model = train_gbt(featured_df)

# 5. Load + predict
loaded_model = load_model("models/gbt_model")
predict(loaded_model, featured_df)

spark.stop()