import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
import requests
from dotenv import load_dotenv
import os
from dotenv import load_dotenv

load_dotenv()
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

# Load model and encoders
ranker = lgb.Booster(model_file='models/ranker_model.txt')
visitor_encoder = joblib.load('models/visitor_encoder.pkl')
product_encoder = joblib.load('models/product_encoder.pkl')

# Load datasets
products_df = pd.read_csv('data/olist_products_dataset.csv')
orders_df = pd.read_csv('data/olist_orders_dataset.csv')
translations_df = pd.read_csv('data/product_category_name_translation.csv')

# Merge translations
products_df = products_df.merge(translations_df, on="product_category_name", how="left")

# Filter valid products
valid_products_df = products_df[products_df['product_id'].isin(product_encoder.classes_)].copy()
valid_products_df['product_id_encoded'] = product_encoder.transform(valid_products_df['product_id'])

# Create mappings
product_id_to_category = dict(zip(
    valid_products_df['product_id_encoded'],
    valid_products_df['product_category_name_english'].str.replace(" ", "_").str.lower()
))
encoded_to_original_product_id = dict(zip(
    valid_products_df['product_id_encoded'],
    valid_products_df['product_id']
))

# Valid visitor IDs
valid_visitor_ids = [vid for vid in orders_df['customer_id'].unique() if vid in visitor_encoder.classes_]

# Pixabay API
PIXABAY_API_URL = 'https://pixabay.com/api/'

def fetch_pixabay_image(category):
    params = {
        'key': PIXABAY_API_KEY,
        'q': str(category).replace("_", " "),
        'image_type': 'photo',
        'per_page': 3,
        'safesearch': 'true'
    }
    try:
        response = requests.get(PIXABAY_API_URL, params=params, timeout=3)
        data = response.json()
        if data.get('hits'):
            return data['hits'][0]['webformatURL']
    except:
        pass
    return "https://via.placeholder.com/120?text=No+Image"

# Streamlit UI
st.title("🎯 Smart Product Ranker")
st.markdown("Select your Visitor ID to get personalized product rankings!")

visitor_id_input = st.selectbox("Select Visitor ID:", options=valid_visitor_ids)

if visitor_id_input:
    try:
        visitor_id_encoded = visitor_encoder.transform([visitor_id_input])[0]

        all_products = pd.DataFrame({
            'visitor_id': [visitor_id_encoded] * len(product_encoder.classes_),
            'product_id': list(range(len(product_encoder.classes_)))
        })

        relevance_scores = ranker.predict(all_products)
        all_products['score'] = relevance_scores

        top_products = all_products.sort_values(by='score', ascending=False).head(10)

        st.subheader("🔥 Top 10 Recommendations")

        # Show image for top product
        top_product = top_products.iloc[0]
        encoded_product_id = top_product['product_id']
        top_category = product_id_to_category.get(encoded_product_id, "unknown")
        image_url = fetch_pixabay_image(top_category)
        st.image(image_url, width=120, caption=f"Top 1: {str(top_category).replace('_', ' ').title()}")

        # Show all product details
        st.subheader("📋 Product Details")

        for i, row in enumerate(top_products.itertuples(), 1):
            encoded_product_id = row.product_id
            score = row.score
            original_product_id = encoded_to_original_product_id.get(encoded_product_id)

            product_row = valid_products_df[valid_products_df['product_id'] == original_product_id]

            category_name = product_id_to_category.get(encoded_product_id, "unknown")
            category_str = str(category_name).replace('_', ' ').title()

            if not product_row.empty:
                product_info = product_row.iloc[0]
                name_length = product_info.get('product_name_lenght', 'N/A')
                desc_length = product_info.get('product_description_lenght', 'N/A')
                weight = product_info.get('product_weight_g', 'N/A')
                length = product_info.get('product_length_cm', 'N/A')
                width = product_info.get('product_width_cm', 'N/A')
                height = product_info.get('product_height_cm', 'N/A')
            else:
                name_length = desc_length = weight = length = width = height = 'N/A'

            st.write(f"**{i}. {category_str}** — Score: `{score:.4f}`")
            st.markdown(f"""
            • Name Length: {name_length if pd.notna(name_length) else 'N/A'}  
            • Description Length: {desc_length if pd.notna(desc_length) else 'N/A'}  
            • Weight: {weight if pd.notna(weight) else 'N/A'}g  
            • Size (L×W×H): {length if pd.notna(length) else 'N/A'}×{width if pd.notna(width) else 'N/A'}×{height if pd.notna(height) else 'N/A'} cm  
            """)

    except ValueError:
        st.error("Invalid Visitor ID selected.")

# Load .env file
load_dotenv()
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

# Don't log your API key in public
# st.write("Loaded API Key:", PIXABAY_API_KEY)  # REMOVE THIS
