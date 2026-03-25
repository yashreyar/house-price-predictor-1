# 🏠 AI House Price Predictor (Random Forest)

An end-to-end Machine Learning application designed to predict residential property prices. This project transitions from raw data exploration to a production-ready web interface.

## 🚀 Live Demo
**https://house-price-predictor-1.streamlit.app/**

## 🏗️ Project Architecture
The project is organized into a modular structure to ensure maintainability and scalability:
- `data/`: Contains the `Housing.csv` dataset.
- `src/`: Core Python scripts for model training and logic.
- `models/`: Serialized Scikit-Learn pipelines (`.pkl` files).
- `notebooks/`: Jupyter Notebooks for Exploratory Data Analysis (EDA).
- `app.py`: Interactive Streamlit frontend.

## 🛠️ Technical Workflow
1. **Exploratory Data Analysis (EDA):** Analyzed correlations using heatmaps. Identified 'Area', 'Bathrooms', and 'Air Conditioning' as primary price drivers.
2. **Feature Engineering:** Implemented **One-Hot Encoding** for categorical variables (Mainroad, Guestroom, Furnishing Status).
3. **Pipeline Integration:** Used a `Scikit-Learn Pipeline` to bundle **StandardScaler** and **Random Forest Regressor**, preventing data leakage.
4. **Hyperparameter Tuning:** Optimized model performance using **GridSearchCV**.
5. **Deployment:** Integrated with GitHub Actions for seamless deployment on **Streamlit Cloud**.

