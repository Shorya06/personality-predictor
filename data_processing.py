import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re

sia = SentimentIntensityAnalyzer()

tfidf = TfidfVectorizer(
    max_features=50,
    stop_words='english',
    ngram_range=(1, 2)
)

def clean_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    except Exception as e:
        print(f"Error in text cleaning: {str(e)}")
        return ""

def extract_linguistic_features(text):
    try:
        text = clean_text(text)
        
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        
        sentiment_scores = sia.polarity_scores(text)
        
        features = {
            'num_words': len(words),
            'num_sentences': len(sentences),
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'sentiment_neg': sentiment_scores['neg'],
            'sentiment_neu': sentiment_scores['neu'],
            'sentiment_pos': sentiment_scores['pos'],
            'sentiment_compound': sentiment_scores['compound']
        }
        
        return features
    except Exception as e:
        print(f"Error in linguistic feature extraction: {str(e)}")
        return {
            'num_words': 0, 'num_sentences': 0,
            'avg_word_length': 0, 'avg_sentence_length': 0,
            'sentiment_neg': 0, 'sentiment_neu': 0,
            'sentiment_pos': 0, 'sentiment_compound': 0
        }

def process_text(text):
    try:
        if not text or pd.isna(text):
            text = ""
        
        text = str(text)
        
        ling_features = extract_linguistic_features(text)
        
        feature_values = list(ling_features.values())
        
        tfidf_features = tfidf.transform([text]).toarray()[0]
        
        combined_features = np.concatenate([feature_values, tfidf_features])
        
        return combined_features
    except Exception as e:
        print(f"Error in text processing: {str(e)}")
        return np.zeros(58)

def process_texts_batch(texts):
    texts = [str(text) if pd.notna(text) else "" for text in texts]
    
    tfidf.fit(texts)
    
    features = []
    for text in texts:
        text_features = process_text(text)
        features.append(text_features)
    
    return np.array(features)

def create_synthetic_dataset(n_samples=1000):
    example_texts = [
        "I love spending time with friends and meeting new people!",
        "I prefer quiet evenings alone with a good book.",
        "Organization and planning are key to success.",
        "I enjoy trying new experiences and adventures.",
        "I tend to worry about small details and future events."
    ]
    
    texts = []
    labels = []
    
    for _ in range(n_samples):
        base_text = np.random.choice(example_texts)
        modified_text = base_text
        
        personality_scores = np.random.rand(5)
        
        texts.append(modified_text)
        labels.append(personality_scores)
    
    return pd.DataFrame({
        'text': texts,
        'openness': [score[0] for score in labels],
        'conscientiousness': [score[1] for score in labels],
        'extraversion': [score[2] for score in labels],
        'agreeableness': [score[3] for score in labels],
        'neuroticism': [score[4] for score in labels]
    })