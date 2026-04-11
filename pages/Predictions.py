def run():
    import streamlit as st
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
    from src.ingestion.market_ingestion import load_stock_data
    from src.preprocessing.feature_engineering_spark import add_features

    st.title("Predictions")

    model = st.session_state.get("model")

    if model is None:
        st.warning("Train model first")
        st.stop()

    spark = SparkSession.builder.getOrCreate()

    # Load data
    pdf = load_stock_data()
    sdf = spark.createDataFrame(pdf)

    # Features
    sdf = add_features(sdf)
    sdf = sdf.dropna()

    # Take latest rows
    latest = sdf.orderBy(col("date").desc()).limit(10)

    # Predict
    predictions = model.transform(latest)

    # Select useful columns
    result = predictions.select(
        "date",
        "price_close",
        "prediction"
    )

    # Show in Streamlit
    st.dataframe(result.toPandas(), use_container_width=True)