import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging

# Setup logging
logging.basicConfig(filename='./logs/training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Load data
df = pd.read_csv('./data/processed/unified_ecommerce_dataset.csv')

# Define feature types
categorical_cols = ['Category', 'SubCategory', 'Brand', 'Material', 'Energy Rating']
numerical_cols = ['RAM (GB)', 'Storage (GB)', 'Screen Size', 'Battery (mAh)', 'Camera (MP)', 'Capacity (Liters)', 'Weight (Kg)']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True)),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='NA', keep_empty_features=True)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50))  # Limit categories
        ]), categorical_cols)
    ])

# Features and target
X = df.drop('Price', axis=1)
y = df['Price'].fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),  # Parallelize
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

best_model = None
best_rmse = float('inf')

for name, model in models.items():
    try:
        model.fit(X_train_preprocessed, y_train)
        y_pred = model.predict(X_test_preprocessed)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"{name}: RMSE = {rmse}, R² = {r2}")
        print(f"{name}: RMSE = {rmse}, R² = {r2}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
    except Exception as e:
        logging.error(f"Error training {name}: {str(e)}")
        print(f"Error training {name}: {str(e)}")

# Save best model and preprocessor
if best_model:
    joblib.dump(best_model, './app/best_model.joblib')
    joblib.dump(preprocessor, './app/preprocessor.joblib')
    logging.info("Best model and preprocessor saved.")
else:
    logging.error("No model was trained successfully.")
    print("No model was trained successfully.")