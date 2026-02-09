import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("models/iris_classifier.pkl")

st.set_page_config(page_title="Sales Predictor", layout="centered")

st.title("📈 Iris Class Prediction App")
st.write("Predict class iris based on lengths and width of petals and sepals.")
st.write("Model expects:", model.coef_.shape[0], "features")

# Input fields
st.subheader("Enter Petal length")
sepal_length = st.number_input("Sepal length", min_value=4.4, max_value= 7.9,step= 0.1)
sepal_width = st.number_input("Sepal length", min_value=2.0, max_value= 4.4,step= 0.1)
petal_length = st.number_input("Petal length", min_value=1.0, max_value= 6.9, step= 0.1)
petal_width = st.number_input("Petal length", min_value=0.1, max_value= 2.5, step= 0.1)

# Predict button
if st.button("Predict Iris Class"):
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(X)[0]
    
    class_map = {
        0: "Iris-setosa",
        1: "Iris-versicolor",
        2: "Iris-virginica"
    }

    species = class_map.get(prediction, "Unknown")

        

    st.success(f"🌼 Predicted Iris Class: **{species}**")

st.markdown("---")
st.caption("Classification model trained on the Iris dataset")
