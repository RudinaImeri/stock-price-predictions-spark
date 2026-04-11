from pyspark.ml import PipelineModel


def load_model(path="models/gbt_model"):
    return PipelineModel.load(path)


def predict(model, df):
    predictions = model.transform(df)
    predictions.select("date", "prediction").show(10)
    return predictions