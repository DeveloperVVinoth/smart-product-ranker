# 🎯 Smart Product Ranker

The **Smart Product Ranker** is a real-time machine learning app that provides personalized product recommendations for users based on their past behavior. It is built with **Streamlit**, powered by a **LightGBM ranking model**, and visualized with category images from the **Pixabay API**.

---

## 🚀 Features

- ✅ Visitor-based personalized product ranking
- ✅ Top 10 product recommendations with scores
- ✅ Displays product features: name length, description length, weight, and size
- ✅ Product category images via Pixabay API
- ✅ Built-in fallback for missing images
- ✅ Clean and interactive UI with Streamlit

---

## 📂 Project Structure

smart-product-ranker/
│
├── app.py # Streamlit app
├── .env # API key (keep secret)
├── .gitignore # Hides sensitive files like .env
├── requirements.txt # Python dependencies
│
├── data/ # Dataset CSV files
│ └── olist_products_dataset.csv
│ └── olist_orders_dataset.csv
│ └── product_category_name_translation.csv
│
├── models/ # Trained ranking model and encoders
│ └── ranker_model.txt
│ └── visitor_encoder.pkl
│ └── product_encoder.pkl
│
├── src/ # Model training scripts
│ └── train_ranker.py
│ └── rank_utils.py



## 📦 Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/smart-product-ranker.git
cd smart-product-ranker

2. Install Dependencies
pip install -r requirements.txt

3. Add Your API Key
Create a .env file and add your Pixabay API key:
PIXABAY_API_KEY=your_pixabay_api_key

🔐 Never share your .env file or API key! The .gitignore already prevents it from being pushed.

4. Run the App
streamlit run app.py

📊 Dataset
This app uses the Brazilian E-Commerce Public Dataset by Olist, which includes detailed order, product, and customer information.

📸 Image Integration
We use the Pixabay API to fetch product category images dynamically based on the top recommendation.

🛡️ Security Note
API keys are stored securely using python-dotenv.
.env is listed in .gitignore and should not be uploaded to GitHub.

🙌 Credits
Dataset: Olist on Kaggle
API: Pixabay
ML Model: LightGBM
App Framework: Streamlit



💡 Future Enhancements
1. Add user search history for more contextual recommendations
2. Use embeddings for deeper product similarity
3. Add product review sentiment analysis

📬 Contact
For questions or suggestions, feel free to reach out!

Let me know if you'd like a version with your GitHub repo link, name, or any more features!
