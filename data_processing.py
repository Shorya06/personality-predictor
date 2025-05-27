import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import re

# Trait-specific keyword dictionaries
TRAIT_KEYWORDS = {
    'openness': [
        'creative', 'curious', 'artistic', 'imaginative', 'innovative', 'adventure',
        'explore', 'learn', 'experience', 'discover', 'unique', 'novel', 'diverse',
        'intellectual', 'philosophical', 'abstract', 'complex'
    ],
    'conscientiousness': [
        'organized', 'responsible', 'disciplined', 'efficient', 'planned', 'thorough',
        'detail', 'systematic', 'careful', 'precise', 'punctual', 'reliable',
        'structured', 'goal', 'achievement', 'focused'
    ],
    'extraversion': [
        'outgoing', 'social', 'energetic', 'enthusiastic', 'active', 'talkative',
        'party', 'group', 'people', 'friends', 'excitement', 'adventure', 'loud',
        'expressive', 'assertive', 'confident'
    ],
    'agreeableness': [
        'kind', 'compassionate', 'cooperative', 'helpful', 'sympathetic', 'warm',
        'considerate', 'friendly', 'generous', 'gentle', 'patient', 'understanding',
        'empathy', 'trust', 'harmony', 'support'
    ],
    'neuroticism': [
        'anxious', 'worried', 'nervous', 'stressed', 'tense', 'moody',
        'emotional', 'sensitive', 'fear', 'doubt', 'overwhelmed', 'concerned',
        'uncertain', 'insecure', 'uncomfortable'
    ]
}

sia = SentimentIntensityAnalyzer()

# Initialize TF-IDF vectorizer with feature names
tfidf = TfidfVectorizer(
    max_features=100,  # Increased for better feature capture
    stop_words='english',
    ngram_range=(1, 3),  # Extended to capture phrases
    analyzer='word'  # Use word analyzer for better interpretability
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

def calculate_trait_scores(text, features):
    """Calculate trait-specific scores based on keywords and linguistic features."""
    text = text.lower()
    words = set(word_tokenize(text))
    
    # Initialize base scores from keyword matches
    trait_scores = {
        trait: sum(1 for keyword in keywords if keyword in text) / len(keywords)
        for trait, keywords in TRAIT_KEYWORDS.items()
    }
    
    # Analyze text complexity
    avg_word_length = features['avg_word_length']
    avg_sentence_length = features['avg_sentence_length']
    
    # Get sentiment and subjectivity
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Adjust trait scores based on linguistic features
    trait_scores['openness'] *= (1 + avg_sentence_length / 20)  # Reward complex sentences
    trait_scores['conscientiousness'] *= (1 + features['num_sentences'] / 10)  # Reward structure
    trait_scores['extraversion'] *= (1 + subjectivity)  # Reward expressiveness
    trait_scores['agreeableness'] *= (1 + (sentiment + 1) / 2)  # Reward positive sentiment
    trait_scores['neuroticism'] *= (1 + abs(sentiment))  # Reward emotional intensity
    
    # Normalize scores
    total = sum(trait_scores.values())
    if total > 0:
        trait_scores = {k: v/total for k, v in trait_scores.items()}
    
    # Enhance contrast between traits
    mean_score = np.mean(list(trait_scores.values()))
    trait_scores = {k: (v - mean_score) * 1.5 + mean_score for k, v in trait_scores.items()}
    
    # Ensure scores are between 0 and 1
    trait_scores = {k: max(0, min(1, v)) for k, v in trait_scores.items()}
    
    return trait_scores

def extract_linguistic_features(text):
    try:
        text = clean_text(text)
        
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        
        sentiment_scores = sia.polarity_scores(text)
        
        # Enhanced feature set
        features = {
            'num_words': len(words),
            'num_sentences': len(sentences),
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'sentiment_neg': sentiment_scores['neg'],
            'sentiment_neu': sentiment_scores['neu'],
            'sentiment_pos': sentiment_scores['pos'],
            'sentiment_compound': sentiment_scores['compound'],
            'text_complexity': avg_word_length * avg_sentence_length / 100,
            'word_variety': len(set(words)) / (len(words) + 1)
        }
        
        return features
    except Exception as e:
        print(f"Error in linguistic feature extraction: {str(e)}")
        return {
            'num_words': 0, 'num_sentences': 0,
            'avg_word_length': 0, 'avg_sentence_length': 0,
            'sentiment_neg': 0, 'sentiment_neu': 0,
            'sentiment_pos': 0, 'sentiment_compound': 0,
            'text_complexity': 0, 'word_variety': 0
        }

def get_feature_names():
    """Get names for all features used in the model."""
    linguistic_features = [
        'word_count',
        'sentence_count',
        'avg_word_length',
        'avg_sentence_length',
        'sentiment_neg',
        'sentiment_neu',
        'sentiment_pos',
        'sentiment_compound',
        'text_complexity',
        'word_variety'
    ]
    
    # Get actual TF-IDF feature names if available
    try:
        tfidf_features = tfidf.get_feature_names_out()
    except:
        tfidf_features = [f'word_pattern_{i}' for i in range(100)]
    
    trait_features = [
        'openness_score',
        'conscientiousness_score',
        'extraversion_score',
        'agreeableness_score',
        'neuroticism_score'
    ]
    
    return linguistic_features + list(tfidf_features) + trait_features

def process_text(text, fit_tfidf=False):
    try:
        if not text or pd.isna(text):
            text = ""
        
        text = str(text)
        
        # Extract linguistic features
        ling_features = extract_linguistic_features(text)
        
        # Calculate trait-specific scores
        trait_scores = calculate_trait_scores(text, ling_features)
        
        # Convert features to array
        feature_values = list(ling_features.values())
        
        # Get TF-IDF features
        if fit_tfidf:
            tfidf.fit([text])
        try:
            tfidf_features = tfidf.transform([text]).toarray()[0]
        except ValueError:
            # If TF-IDF is not fitted, return zero vector
            tfidf_features = np.zeros(100)  # max_features size
        
        # Combine all features
        combined_features = np.concatenate([
            feature_values,
            tfidf_features,
            list(trait_scores.values())
        ])
        
        return combined_features
    except Exception as e:
        print(f"Error in text processing: {str(e)}")
        return np.zeros(115)  # Adjusted feature size

def process_texts_batch(texts):
    texts = [str(text) if pd.notna(text) else "" for text in texts]
    
    # Fit TF-IDF on all texts and store feature names
    tfidf.fit(texts)
    
    features = []
    for text in texts:
        text_features = process_text(text)  # No need to fit TF-IDF here
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