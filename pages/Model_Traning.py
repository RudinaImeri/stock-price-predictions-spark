def run():
    import streamlit as st
    from pyspark.sql import SparkSession
    from src.ingestion.market_ingestion import load_stock_data
    from src.preprocessing.feature_engineering_spark import add_features
    from src.models.train_spark import train_model

    st.title("Model Training (Spark)")

    spark = SparkSession.builder.getOrCreate()

    FEATURES = [
        "ret_1",
        "ret_2",
        "ret_3",
        "volatility_3",
        "sma_ratio",
        "momentum_5",
    ]

    if st.button("Train Model"):

        with st.spinner("Training model... please wait ⏳"):

            # Load data
            pdf = load_stock_data()
            sdf = spark.createDataFrame(pdf)

            # Feature engineering
            sdf = add_features(sdf)
            sdf = sdf.dropna()

            # Train model
            model = train_model(sdf, FEATURES)

            st.session_state["model"] = model
            st.session_state["features"] = FEATURES

        st.success("✅ Model trained successfully!")