import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# Load the Pipeline (Scaler + Model bundled together)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "house_pipeline.pkl")

# Check if model exists before loading
if os.path.exists(MODEL_PATH):
    pipeline = joblib.load(MODEL_PATH)
else:
    st.error("Model file not found! Please run 'python src/train.py' first.")

st.title("🏠 AI House Price Predictor")
st.markdown("Predicting property value using **Random Forest** and **One-Hot Encoding**.")

st.divider()

# --- Input Section ---
col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (sq ft)", min_value=100, max_value=20000, value=5000)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Bathrooms", 1, 4, 2)
    stories = st.slider("Stories", 1, 4, 1)
    parking = st.slider("Parking", 0, 3, 1)

with col2:
    mainroad = st.selectbox("Main Road?", ["Yes", "No"])
    guestroom = st.selectbox("Guestroom?", ["Yes", "No"])
    basement = st.selectbox("Basement?", ["Yes", "No"])
    ac = st.selectbox("Air Conditioning?", ["Yes", "No"])
    prefarea = st.selectbox("Preferred Area?", ["Yes", "No"])
    furnishing = st.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])

# --- Prediction Logic ---
if st.button("Predict Price", use_container_width=True):
    try:
        # Construct input dataframe to match training columns exactly
        input_data = pd.DataFrame([{
            'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms, 
            'stories': stories, 'parking': parking,
            'mainroad_yes': 1 if mainroad == "Yes" else 0,
            'guestroom_yes': 1 if guestroom == "Yes" else 0,
            'basement_yes': 1 if basement == "Yes" else 0,
            'hotwaterheating_yes': 0, 
            'airconditioning_yes': 1 if ac == "Yes" else 0,
            'prefarea_yes': 1 if prefarea == "Yes" else 0,
            'furnishingstatus_semi-furnished': 1 if furnishing == "Semi-Furnished" else 0,
            'furnishingstatus_unfurnished': 1 if furnishing == "Unfurnished" else 0
        }])

        # Predict using pipeline (Scaling is handled automatically inside!)
        prediction = pipeline.predict(input_data)[0]
        st.success(f"💰 Estimated Price: ₹ {prediction:,.2f}")

        # Show Feature Importance Chart
        st.subheader("🔍 Model Decision Drivers")
        rf_model = pipeline.named_steps['rf']
        importance_df = pd.DataFrame({
            "Feature": input_data.columns,
            "Importance": rf_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Feature"))

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

st.divider()
st.markdown("Built with ❤️ | Data & AI Intelligence Project")