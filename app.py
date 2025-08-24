
import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai

# Load model and preprocessor
model = joblib.load('./app/best_model.joblib')
preprocessor = joblib.load('./app/preprocessor.joblib')

# Gemini setup
genai.configure(api_key='AIzaSyATFMGFyf5UxUaP7C_qJq1gzbyT97SQwMk')  # Replace with your key
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Streamlit UI
st.title('E-commerce Price Prediction Tool')

# Category selection with examples
category_options = {
    'Laptop': 'Electronics for computing (e.g., Apple, HP)',
    'Mobile': 'Smartphones (e.g., Apple, Samsung)',
    'Appliance': 'Home appliances like TVs, refrigerators (e.g., Samsung, LG)',
    'Shoes': 'Footwear (e.g., Adidas, Nike)',
    'Furniture': 'Home furniture (e.g., Pepperfry, IKEA)',
    'Clothes': 'Apparel (e.g., Adidas, H&M)'
}
category = st.selectbox('Category', list(category_options.keys()), help='Select the product type.')
st.info(category_options.get(category, ''))

# Brand options based on dataset
brand_options = {
    'Laptop': ['Apple', 'HP', 'Lenovo', 'Asus', 'Dell'],
    'Mobile': ['Apple', 'Samsung', 'Realme', 'OnePlus', 'Xiaomi'],
    'Appliance': ['Panasonic', 'Philips', 'Whirlpool', 'Samsung', 'LG'],
    'Shoes': ['Adidas', 'Puma', 'Bata', 'Nike', 'Reebok'],
    'Furniture': ['Pepperfry', 'IKEA', 'UrbanLadder', 'Godrej'],
    'Clothes': ['Adidas', 'Nike', 'Levis', 'H&M', 'Zara']
}
brand = st.selectbox('Brand', brand_options[category] + ['Other'], help='Select brand or choose Other for custom input.')
if brand == 'Other':
    brand = st.text_input('Custom Brand', 'Unknown')

subcategory = st.text_input('SubCategory (e.g., Model name)', 'General')

# Dynamic inputs based on category
if category in ['Laptop', 'Mobile']:
    ram = st.number_input('RAM (GB)', min_value=0.0, value=4.0, step=1.0, help='e.g., 4, 8, 16, 32')
    storage = st.number_input('Storage (GB)', min_value=0.0, value=128.0, step=64.0, help='e.g., 128, 256, 512, 1024')
    cpu = st.selectbox('CPU/Processor', ['i3', 'i5', 'i7', 'Ryzen 5', 'Ryzen 7', 'Snapdragon 888', 'Snapdragon 720G', 'A14 Bionic', 'Other'], help='Select processor or choose Other.')
    if cpu == 'Other':
        cpu = st.text_input('Custom CPU/Processor', 'NA')
    screen_size = st.number_input('Screen Size (inches)', min_value=0.0, value=13.3, step=0.1, help='e.g., 13.3, 15.6')
else:
    ram = 0.0
    storage = 0.0
    cpu = 'NA'
    screen_size = 0.0

if category == 'Mobile':
    battery = st.number_input('Battery (mAh)', min_value=0, value=3000, step=500, help='e.g., 3000, 4000, 5000')
    camera = st.number_input('Camera (MP)', min_value=0, value=12, step=1, help='e.g., 12, 48, 64')
else:
    battery = 0
    camera = 0

if category == 'Appliance':
    capacity = st.number_input('Capacity (Liters)', min_value=0.0, value=0.0, step=1.0, help='e.g., 200 for refrigerators')
    energy_rating = st.selectbox('Energy Rating', ['NA', '3 Star', '4 Star', '5 Star'], help='Energy efficiency rating.')
else:
    capacity = 0.0
    energy_rating = 'NA'

weight = st.number_input('Weight (Kg)', min_value=0.0, value=0.0, step=0.1, help='e.g., 1.5 for laptops, 50 for furniture')
material = st.selectbox('Material', ['NA', 'Cotton', 'Silk', 'Denim', 'Polyester', 'Leather', 'Synthetic', 'Canvas', 'Wood', 'Metal', 'Plastic'], help='Select material.')
features = st.text_input('Features (e.g., Size, Color, Dimensions)', 'NA', help='e.g., Size: M, Color: Black, Dimensions: 108x55x181')

if st.button('Predict Price'):
    input_data = pd.DataFrame({
        'Category': [category], 'SubCategory': [subcategory], 'Brand': [brand],
        'RAM (GB)': [ram], 'Storage (GB)': [storage], 'CPU/Processor': [cpu], 'GPU': ['NA'],
        'Screen Size': [screen_size], 'Battery (mAh)': [battery], 'Camera (MP)': [camera],
        'Capacity (Liters)': [capacity], 'Energy Rating': [energy_rating],
        'Weight (Kg)': [weight], 'Material': [material], 'Features': [features]
    })
    
    input_preprocessed = preprocessor.transform(input_data)
    prediction = model.predict(input_preprocessed)[0]
    st.success(f'Predicted Price: Rs.{prediction:.2f}')
    
    prompt = f"User input: {input_data.to_dict()}. Predicted price: Rs. {prediction}. Explain in plain language why this price might be unusually high or low or perfect based on the specs, or what features drive it."
    try:
        response = gemini_model.generate_content(prompt)
        st.write('Explanation:', response.text)
    except Exception as e:
        st.error(f"Gemini API error: {str(e)}")