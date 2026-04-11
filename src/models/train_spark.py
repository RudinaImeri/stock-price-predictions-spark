from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline


def train_model(df, feature_cols):

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    gbt = GBTClassifier(
        labelCol="result",
        featuresCol="features",
        maxIter=50
    )

    pipeline = Pipeline(stages=[assembler, gbt])

    return pipeline.fit(df)