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
from textblob import TextBlob

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

def create_trait_radar_chart(trait_values, traits=None):
    if traits is None:
        traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    values_pct = [v * 100 for v in trait_values]
    max_idx = np.argmax(values_pct)
    colors = ['#5eead4'] * len(traits)
    colors[max_idx] = '#facc15'  # Highlight color (yellow)
    marker_symbols = ['circle'] * len(traits)
    marker_symbols[max_idx] = 'star'
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_pct,
        theta=traits,
        fill='toself',
        fillcolor='rgba(94,234,212,0.2)',
        line=dict(color='#7f9cf5', width=4, shape='spline'),
        marker=dict(size=18, color=colors, symbol=marker_symbols, line=dict(width=3, color='#fff')),
        text=[f"{v:.1f}%" for v in values_pct],
        textposition='top center',
        mode='markers+lines+text',
        name='Traits'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=16, color='#e6e6f0')),
            angularaxis=dict(tickfont=dict(size=18, color='#e6e6f0'))
        ),
        showlegend=False,
        margin=dict(t=40, b=40, l=40, r=40),
        paper_bgcolor='#181c2f',
        plot_bgcolor='#181c2f',
        font=dict(family='Inter, Poppins, sans-serif', size=18, color='#e6e6f0'),
        height=500
    )
    fig.add_annotation(
        text=f"⭐ Dominant: {traits[max_idx]} ({values_pct[max_idx]:.1f}%)",
        x=0.5, y=1.2, showarrow=False, font=dict(size=20, color='#facc15'),
        xref="paper", yref="paper", align="center"
    )
    return fig

def create_trait_bar_chart(trait_values, traits=None):
    if traits is None:
        traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    values_pct = [v * 100 for v in trait_values]
    max_idx = np.argmax(values_pct)
    colors = ['#7f9cf5'] * len(traits)
    colors[max_idx] = '#facc15'
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=traits, y=values_pct, marker_color=colors,
        text=[f'{v:.1f}%' for v in values_pct], textposition='outside',
        marker_line_width=2, marker_line_color='#181c2f',
    ))
    fig.update_layout(
        yaxis=dict(title='Trait Score', ticksuffix='%', range=[0, 100], tickfont=dict(size=16)),
        xaxis=dict(tickfont=dict(size=16)),
        margin=dict(t=60, b=40, l=40, r=40),
        paper_bgcolor='#181c2f',
        plot_bgcolor='#181c2f',
        font=dict(family='Inter, Poppins, sans-serif', size=18, color='#e6e6f0'),
        height=400,
        showlegend=False
    )
    fig.add_annotation(
        text=f"⭐ Dominant: {traits[max_idx]} ({values_pct[max_idx]:.1f}%)",
        x=0.5, y=1.1, showarrow=False, font=dict(size=20, color='#facc15'),
        xref="paper", yref="paper", align="center"
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