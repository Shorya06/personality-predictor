# Personality Trait Predictor

This project uses machine learning and NLP to predict Big Five personality traits from text input. It analyzes social media-style text to predict Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism scores.

## Features

- Text analysis and personality prediction
- Statistical analysis of input text
- Interactive visualizations
- BERT-based text embeddings
- Machine learning model predictions

## Setup

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit application
- `data_processing.py`: Text processing and feature extraction
- `model.py`: Machine learning model implementation
- `utils.py`: Utility functions and helpers
- `data/`: Directory for dataset storage
- `models/`: Directory for saved model files

## Usage

1. Enter text in the input box on the Streamlit interface
2. Click "Predict" to analyze the text
3. View personality predictions and visualizations

## Model Details

The system uses BERT embeddings for text vectorization and a Random Forest classifier for prediction. Features include:
- Text statistical analysis
- Semantic embeddings
- Linguistic features
- Sentiment analysis 