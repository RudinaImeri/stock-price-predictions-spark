def run():
    import streamlit as st
    import pandas as pd

    st.title("Feature Importance")

    model = st.session_state.get("model")
    features = st.session_state.get("features")

    if model is None:
        st.warning("Train model first")
        st.stop()

    # GBT is last stage in pipeline
    gbt_model = model.stages[-1]

    importances = gbt_model.featureImportances.toArray()

    df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("Feature"))