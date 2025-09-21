import streamlit as st
import pandas as pd
import joblib

st.title("NYC Airbnb Price Predictor")

@st.cache_resource
def load_model():
    return joblib.load("models/price_model.joblib.xz")

pipe = load_model()

# UI
neighbourhood_group = st.selectbox("Neighbourhood Group", ["Manhattan","Brooklyn","Queens","Bronx","Staten Island"])
room_type = st.selectbox("Room Type", ["Entire home/apt","Private room","Shared room"])
minimum_nights = st.number_input("Minimum Nights", 1, 365, 3)
number_of_reviews = st.number_input("Number of Reviews", 0, 1000, 10)
reviews_per_month = st.number_input("Reviews per Month", 0.0, 30.0, 1.0)
calculated_host_listings_count = st.number_input("Host Listings Count", 0, 500, 1)
availability_365 = st.number_input("Availability (days/year)", 0, 365, 180)

df_in = pd.DataFrame([{
    "neighbourhood_group": neighbourhood_group,
    "room_type": room_type,
    "minimum_nights": float(minimum_nights),
    "number_of_reviews": float(number_of_reviews),
    "reviews_per_month": float(reviews_per_month),
    "calculated_host_listings_count": float(calculated_host_listings_count),
    "availability_365": float(availability_365),
}])

prediction = pipe.predict(df_in)[0]
st.success(f"Predicted Price: ${prediction:.2f} per night")
