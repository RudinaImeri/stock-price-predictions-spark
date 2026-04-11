from pyspark.sql import functions as F
from pyspark.sql.window import Window


def add_features(df):

    w = Window.partitionBy("symbol").orderBy("date")

    df = df.withColumn(
        "ret_1",
        F.col("price_close") /
        F.lag("price_close", 1).over(w) - 1
    )

    df = df.withColumn(
        "ret_2",
        F.col("price_close") /
        F.lag("price_close", 2).over(w) - 1
    )

    df = df.withColumn(
        "ret_3",
        F.col("price_close") /
        F.lag("price_close", 3).over(w) - 1
    )

    df = df.withColumn(
        "volatility_3",
        F.stddev("price_close").over(w.rowsBetween(-3, 0))
    )

    df = df.withColumn(
        "sma_5",
        F.avg("price_close").over(w.rowsBetween(-5, 0))
    )

    df = df.withColumn(
        "sma_ratio",
        F.col("price_close") / F.col("sma_5")
    )

    df = df.withColumn(
        "momentum_5",
        F.col("price_close") -
        F.lag("price_close", 5).over(w)
    )

    # Label (binary classification: up/down)
    df = df.withColumn(
        "result",
        F.when(
            F.lead("price_close", 1).over(w) >
            F.col("price_close"),
            1
        ).otherwise(0)
    )

    df = df.dropna()

    return df
