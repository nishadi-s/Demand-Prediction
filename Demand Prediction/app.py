import streamlit as st
import pandas as pd
from model import train_and_save_model, predict_new_input

# Streamlit app title
st.title("Sales Quantity Prediction")

# Cache the model training to avoid retraining
@st.cache_resource
def load_model():
    model, scaler, features = train_and_save_model()
    return model, scaler, features

# Load model, scaler, and features
model, scaler, features = load_model()

# Create a form for input
st.header("Enter Input Values")
with st.form(key="prediction_form"):
    # Numeric inputs
    price = st.number_input("Price ($)", min_value=0.0, value=10.5, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, value=22.0, step=0.1)
    year = st.number_input("Year", min_value=2000, value=2023, step=1, format="%d")
    sales_lag_1 = st.number_input("Sales Lag 1 (Previous Day)", min_value=0.0, value=105.0, step=0.1)
    sales_lag_7 = st.number_input("Sales Lag 7 (7 Days Ago)", min_value=0.0, value=100.0, step=0.1)

    # Categorical inputs (dropdowns)
    promotion = st.selectbox("Promotion (0 = No, 1 = Yes)", [0, 1], index=1)
    holiday = st.selectbox("Holiday (0 = No, 1 = Yes)", [0, 1], index=0)
    is_weekend = st.selectbox("Is Weekend (0 = No, 1 = Yes)", [0, 1], index=0)
    day_of_week = st.selectbox("Day of Week (0 = Monday, 6 = Sunday)", list(range(7)), index=2)
    month = st.selectbox("Month", list(range(1, 13)), index=5)  # 5 for June (month=6)
    quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=1)  # 1 for Q2

    # Submit button
    submit_button = st.form_submit_button(label="Predict")

    # Process form submission
    if submit_button:
        # Create input dictionary
        input_data = {
            'price': price,
            'temperature': temperature,
            'promotion': promotion,
            'holiday': holiday,
            'day_of_week': day_of_week,
            'month': month,
            'year': year,
            'is_weekend': is_weekend,
            'quarter': quarter,
            'sales_lag_1': sales_lag_1,
            'sales_lag_7': sales_lag_7
        }

        # Make prediction
        prediction = predict_new_input(model, scaler, features, input_data)

        # Display results
        st.subheader("Prediction Results")
        st.write("**Input Values:**")
        st.json(input_data)
        st.write(f"**Predicted Sales Quantity:** {prediction:.2f}")

# Optional: Display model metrics
st.sidebar.header("Model Information")
st.sidebar.write("Model: Random Forest Regressor")
st.sidebar.write("Trained on synthetic sales data")
st.sidebar.write("Metrics available in console (MSE ~30.25, R² ~0.85)")