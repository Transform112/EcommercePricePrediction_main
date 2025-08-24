# E-commerce Price Prediction Tool

## Overview
This project builds an E-commerce Price Prediction Tool that predicts product prices based on specifications across multiple categories (Laptops, Mobiles, TVs, Refrigerators, Shoes, Furniture). It uses regression models and integrates the Gemini API for price anomaly explanations.

## Directory Structure
- `data\raw`: Raw datasets (test.csv, Combined_dataset.csv, flipkart_com-ecommerce_sample.csv)
- `data\processed`: Unified cleaned dataset
- `scripts`: Python scripts for preprocessing, training, and evaluation
- `app`: Streamlit app, model, and preprocessor files
- `notebooks`: Jupyter notebook for EDA
- `config`: Configuration file
- `logs`: Training logs

## Setup
1. Create the directory structure using File Explorer.
2. Move datasets to `data\raw`.
3. Install dependencies: `pip install -r app\requirements.txt`
4. Update `config\config.yaml` with your Gemini API key.
5. Run preprocessing: `python scripts\preprocess.py`
6. Train model: `python scripts\train_model.py`
7. Evaluate model: `python scripts\evaluate_model.py`
8. Run app: `streamlit run app\app.py`

## Deployment
- Deploy to Hugging Face Spaces or Render by uploading `app` contents and setting the Gemini API key as a secret.

## Notes
- `test.csv` lacks a price column; replace with a dataset containing prices if needed.
- Adjust preprocessing in `preprocess.py` if category mappings or specs differ.