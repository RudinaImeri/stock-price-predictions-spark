def run():
    import streamlit as st
    from pyspark.sql import SparkSession
    from src.ingestion.market_ingestion import load_stock_data
    from src.preprocessing.feature_engineering_spark import add_features

    st.title("Data Overview")

    spark = SparkSession.builder.getOrCreate()

    pdf = load_stock_data()
    sdf = spark.createDataFrame(pdf)

    sdf = add_features(sdf)

    st.dataframe(sdf.limit(20).toPandas())