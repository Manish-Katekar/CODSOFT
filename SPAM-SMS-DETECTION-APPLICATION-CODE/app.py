import streamlit as st
import joblib
import re
import string
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import os

os.chdir("C:\\Users\\manis\\OneDrive\\Desktop\\python exercises\\project_NLP\\spam_ml")

vectorizer = joblib.load('C:\\Users\\manis\\OneDrive\\Desktop\\python exercises\\project_NLP\\spam_ml\\vectors\\my_vectorizer.pickel')
spam_clf = joblib.load('C:\\Users\\manis\\OneDrive\\Desktop\\python exercises\\project_NLP\\spam_ml\\models\\my_spam_model.pkl')

def preprocess_and_vectorize_text(text):
    # Define text preprocessing steps (same as before)
    def remove_stopwords(text):
        stop_words = set(stopwords.words("english"))
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        return " ".join(filtered_tokens)

    def lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(lemmatized_tokens)

    def remove_special_characters(text):
        text = re.sub(r'\d+', ' ', text)  # Remove digits
        text = text.lower()  # Convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        return text

    # Apply preprocessing steps to the input text
    preprocessed_text = remove_special_characters(text)
    preprocessed_text = remove_stopwords(preprocessed_text)
    preprocessed_text = lemmatize_text(preprocessed_text)

    vectorized_text = vectorizer.transform([preprocessed_text])
    return vectorized_text

def main(title="Spam_SMS_Detection APP".upper()):
    st.markdown(f"<h1 style='text-align: center; font-size: 25px; color: blue;'>{title}</h1>", unsafe_allow_html=True)
    st.image("C:\\Users\\manis\\OneDrive\\Desktop\\python exercises\\project_NLP\\spam_ml\\images\\image.jpg", width=100)

    with st.expander("1. Check if your text is spam or not"):
        text_message = st.text_input("Please enter your message")
        vectorized_text = preprocess_and_vectorize_text(text_message)  # Don't pass vectorizer
        

        if st.button("Predict"):
            prediction = spam_clf.predict(vectorized_text)

            if prediction[0] == 0:
                info = 'NOT SPAM'
            else:
                info = 'SPAM'
            st.success('Prediction: {}'.format(info))

if __name__ == "__main__":
    main()
