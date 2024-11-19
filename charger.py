
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Title and introduction
st.title("Sentiment Analysis for Retail and E-Commerce Reviews")
st.write("This app predicts sentiment (positive/negative) for e-commerce reviews.")

# Load the dataset (Assume the dataset is preprocessed or already available in the environment)
@st.cache_data
def load_data():
    # For demonstration, I'm creating a mock dataset. Replace this with actual data loading.
    data = {'review': ['Great product', 'Worst service', 'Loved it', 'Will not recommend', 'Good quality'],
            'sentiment': [1, 0, 1, 0, 1]}  # 1 for positive, 0 for negative
    return pd.DataFrame(data)

df = load_data()

# Preprocess the data and split into train/test
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['review']).toarray()
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy display
y_pred = model.predict(X_test)
st.write("Model accuracy:", accuracy_score(y_test, y_pred))

# Input section for user to try predictions
st.header("Try the Sentiment Predictor")
user_input = st.text_area("Enter a product review:")

if st.button("Predict"):
    if user_input:
        # Transform the user input into the same TF-IDF space as training data
        user_input_tfidf = tfidf.transform([user_input])
        prediction = model.predict(user_input_tfidf)[0]

        # Display the result
        if prediction == 1:
            st.success("Positive sentiment detected!")
        else:
            st.error("Negative sentiment detected!")
    else:
        st.warning("Please enter some text for prediction.")
