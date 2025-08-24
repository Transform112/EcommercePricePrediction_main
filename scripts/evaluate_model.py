import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging

# Setup logging
logging.basicConfig(filename='./logs/training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Load data
df = pd.read_csv('./data/processed/unified_ecommerce_dataset.csv')
X = df.drop('Price', axis=1)
y = df['Price']

# Load model and preprocessor
model = joblib.load('./app/best_model.joblib')
preprocessor = joblib.load('./app/preprocessor.joblib')

# Preprocess
X_preprocessed = preprocessor.transform(X)

# Evaluate
y_pred = model.predict(X_preprocessed)
rmse = mean_squared_error(y, y_pred, squared=False)
r2 = r2_score(y, y_pred)
logging.info(f"Evaluation - RMSE: {rmse}, R²: {r2}")
print(f"Evaluation - RMSE: {rmse}, R² = {r2}")