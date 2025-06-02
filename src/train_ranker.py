import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib

from sklearn.preprocessing import LabelEncoder

# Load datasets
orders = pd.read_csv('data/olist_orders_dataset.csv')
order_items = pd.read_csv('data/olist_order_items_dataset.csv')

# Merge to get one row per product per order
df = pd.merge(order_items, orders, on='order_id')

# Drop NA rows (for simplicity)
df.dropna(inplace=True)

# Encode visitor_id (customer_id) and product_id
visitor_encoder = LabelEncoder()
product_encoder = LabelEncoder()

df['visitor_id'] = visitor_encoder.fit_transform(df['customer_id'])
df['product_id'] = product_encoder.fit_transform(df['product_id'])

# Create target label: days to deliver (as a ranking goal)
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])

df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
df = df[df['delivery_days'] > 0]

# Features and target
X = df[['visitor_id', 'product_id']]

# Invert delivery days (lower days = higher relevance)
df['relevance'] = pd.qcut(df['delivery_days'], q=4, labels=False, duplicates='drop')
y = df['relevance'].astype(int)


# Group data by visitor (so the model knows which products were shown to whom)
group = df.groupby('visitor_id').size().values

# Train the model
ranker = lgb.LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    boosting_type='gbdt',
    verbose=1
)

ranker.fit(X, y, group=group)

# Save model and encoders
os.makedirs('models', exist_ok=True)
ranker.booster_.save_model('models/ranker_model.txt')
joblib.dump(visitor_encoder, 'models/visitor_encoder.pkl')
joblib.dump(product_encoder, 'models/product_encoder.pkl')

print("✅ Model training complete and saved!")