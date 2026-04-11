from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def train_gbt(df):

    # Features
    feature_cols = [
        "ret_1", "ret_2", "ret_3",
        "volatility_3",
        "sma_ratio",
        "momentum_5"
    ]

    # Assemble features into vector
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    # Model
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="result",
        maxIter=20
    )

    # Pipeline
    pipeline = Pipeline(stages=[assembler, gbt])

    # Train / test split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Train model
    model = pipeline.fit(train_df)

    # Predictions
    predictions = model.transform(test_df)

    # Evaluate
    evaluator = MulticlassClassificationEvaluator(
        labelCol="result",
        predictionCol="prediction",
        metricName="accuracy"
    )

    accuracy = evaluator.evaluate(predictions)

    print(f"✅ Accuracy: {accuracy:.4f}")

    predictions.select("result", "prediction").show(10)

    model.write().overwrite().save("models/gbt_model")
    
    return model
