import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('vader_lexicon')

def analyze_text_statistics(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    stats = {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'unique_words': len(set(words))
    }
    return stats

def create_trait_radar_chart(predictions):
    categories = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=predictions,
        theta=categories,
        fill='toself',
        name='Personality Traits'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title='Big Five Personality Traits Prediction'
    )
    return fig

def create_trait_bar_chart(predictions):
    categories = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    
    fig = px.bar(
        x=categories,
        y=predictions,
        title='Personality Traits Scores',
        labels={'x': 'Trait', 'y': 'Score'},
        color=predictions,
        color_continuous_scale='Viridis'
    )
    return fig

def generate_wordcloud(text):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

def plot_feature_importance(feature_names, importance_scores, top_n=10):
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(top_n)
    
    fig = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} Most Important Features',
        labels={'importance': 'Importance Score', 'feature': 'Feature'}
    )
    return fig

def analyze_email_metrics(text):
    sia = SentimentIntensityAnalyzer()
    
    sentiment_scores = sia.polarity_scores(text)
    sentiment_score = sentiment_scores['compound']
    
    formality_indicators = {
        'greetings': ['dear', 'hello', 'hi', 'greetings'],
        'closings': ['sincerely', 'regards', 'best', 'thanks', 'thank you'],
        'formal_words': ['please', 'kindly', 'appreciate', 'request', 'inform']
    }
    
    text_lower = text.lower()
    
    formality_count = sum(
        1 for category in formality_indicators.values()
        for word in category if word in text_lower
    )
    
    formality_score = min(1.0, formality_count / 5.0)
    
    urgent_phrases = ['urgent', 'asap', 'as soon as possible', 'emergency', 'immediate']
    has_urgent_phrases = any(phrase in text_lower for phrase in urgent_phrases)
    
    response_urgency = 'High' if has_urgent_phrases else 'Normal'
    
    return {
        'formality_score': formality_score,
        'sentiment_score': sentiment_score,
        'response_urgency': response_urgency
    }