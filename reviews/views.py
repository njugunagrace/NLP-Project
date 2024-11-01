import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from django.conf import settings
from django.shortcuts import render

# Load models and vectorizer
models = {
    'Naive Bayes': joblib.load(os.path.join(settings.BASE_DIR, 'models/nb_model.pkl')),
    'Logistic Regression': joblib.load(os.path.join(settings.BASE_DIR, 'models/log_reg_model.pkl')),
    'Decision Tree': joblib.load(os.path.join(settings.BASE_DIR, 'models/dt_model.pkl'))
}

tfidf_vectorizer = joblib.load(os.path.join(settings.BASE_DIR, 'models/vectorizer.pkl'))

# Download stopwords
nltk.download('stopwords')

# Extend default stop words with NLTK's list
extended_stop_words = ENGLISH_STOP_WORDS.union(stopwords.words('english'))

# Preprocess review by cleaning and vectorizing
def preprocess_review(review):
    # Clean text: lowercase, remove non-alphabetic characters
    review = review.lower()
    review = re.sub(r'[^a-z\s]', '', review)

    # Remove stopwords
    review_tokens = review.split()
    cleaned_review_tokens = [word for word in review_tokens if word not in extended_stop_words]
    cleaned_review = ' '.join(cleaned_review_tokens)

    # Vectorize the cleaned text
    return tfidf_vectorizer.transform([cleaned_review])

# View to handle review input and return sentiment prediction and confidence
def predict_sentiment(request):
    sentiment = None  
    user_review = ''  
    confidence_level = None  

    if request.method == 'POST':
        user_review = request.POST.get('review', '')  # Get the review text from the user
        processed_review = preprocess_review(user_review)  # Clean and vectorize the review

        # Predict sentiment using logistic regression
        for model in models.items():
            predicted_label = model.predict(processed_review)[0]
        # Calculate the confidence level using predict_proba
        try:
                confidence_score = model.predict_proba(processed_review)[0]
                confidence_level = round(max(confidence_score) * 100, 2)  # Confidence in percentage
        except AttributeError:
                confidence_level = 'N/A'
        # Set the sentiment based on the predicted label
        sentiment = 'Positive' if predicted_label == 1 else 'Negative'

    return render(request, 'reviews/index.html', {
        'sentiment': sentiment,
        'review': user_review,
        'confidence_level': confidence_level  
    })
