import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Dynamic Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "housing.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load & Encode Categorical Data (AC, Furnishing, etc.)
data = pd.read_csv(DATA_PATH)
data_encoded = pd.get_dummies(data, drop_first=True)

X = data_encoded.drop("price", axis=1)
y = data_encoded["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build Pipeline: Scaler + Random Forest
# Using a Pipeline ensures new data is scaled exactly like training data
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42))
])

# 3. Hyperparameter Tuning (Shows you are an advanced AI student)
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5]
}

print("Starting Grid Search... this may take a moment.")
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

# 4. Save the Final Optimized Pipeline
best_pipeline = grid.best_estimator_
joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "house_pipeline.pkl"))

# 5. Output Final Metric for your Interview
final_r2 = r2_score(y_test, best_pipeline.predict(X_test))
print(f"--- SUCCESS ---")
print(f"Final Model R2 Score: {final_r2:.4f}")
print(f"Pipeline saved to: {os.path.join(MODEL_DIR, 'house_pipeline.pkl')}")