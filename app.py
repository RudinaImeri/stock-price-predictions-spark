import streamlit as st

import os, sys
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

st.title("📈 Stock Prediction App")

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Overview",
    "Model Training",
    "Feature Importance",
    "Predictions"
])

with tab1:
    from pages.Data_Overview import run
    run()

with tab2:
    from pages.Model_Traning import run
    run()

with tab3:
    from pages.Feature_Importance import run
    run()

with tab4:
    from pages.Predictions import run
    run()