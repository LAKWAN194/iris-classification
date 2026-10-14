from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

# Load trained model (the scaler is part of it)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "iris_classifier.pkl"
model = joblib.load(MODEL_PATH)

CLASS_MAP = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}

st.set_page_config(page_title="Iris Classifier", layout="centered")

st.title("🌸 Iris Class Prediction App")
st.write("Predict the species of an iris from the length and width of its petals and sepals.")

# Input fields
st.subheader("Enter measurements (cm)")
sepal_length = st.number_input("Sepal length", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
sepal_width = st.number_input("Sepal width", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
petal_length = st.number_input("Petal length", min_value=1.0, max_value=7.0, value=4.3, step=0.1)
petal_width = st.number_input("Petal width", min_value=0.1, max_value=2.5, value=1.3, step=0.1)

# Predict button
if st.button("Predict Iris Class"):
    X = pd.DataFrame({
        "sepal_length": [sepal_length],
        "sepal_width": [sepal_width],
        "petal_length": [petal_length],
        "petal_width": [petal_width]
    })

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    species = CLASS_MAP.get(prediction, "Unknown")

    st.success(f"🌼 Predicted Iris Class: **{species}**")

    st.write("Confidence for each class:")
    st.bar_chart(pd.Series(probabilities, index=list(CLASS_MAP.values())))

st.markdown("---")
st.caption("Classification model trained on the Iris dataset")
