import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/wine_model.pkl")

st.title("🍷 Wine Quality Classifier")
st.write("Predict whether a wine is 'Good' (quality ≥ 7) or 'Not Good'.")

# User inputs
fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, 8.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.6, 0.5)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.number_input("Residual Sugar", 0.5, 15.5, 2.5)
chlorides = st.number_input("Chlorides", 0.01, 0.61, 0.08)
free_so2 = st.number_input("Free Sulfur Dioxide", 1, 72, 15)
total_so2 = st.number_input("Total Sulfur Dioxide", 6, 289, 46)
density = st.number_input("Density", 0.99, 1.01, 1.0)
pH = st.number_input("pH", 2.9, 4.0, 3.3)
sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.6)
alcohol = st.number_input("Alcohol", 8.0, 15.0, 10.0)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_so2, total_so2, density, pH, sulphates, alcohol
    ]], columns=[
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol"
    ])
    
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ Good Quality Wine!")
    else:
        st.error("❌ Not a Good Quality Wine")
