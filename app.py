import streamlit as st
import numpy as np
import pandas as pd
from model import PersonalityPredictor
from utils import (
    analyze_text_statistics,
    create_trait_radar_chart,
    create_trait_bar_chart,
    generate_wordcloud,
    plot_feature_importance,
    analyze_email_metrics
)

st.set_page_config(
    page_title="Text Personality Analyzer",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = PersonalityPredictor()
    try:
        # Load pre-trained model
        with st.spinner("Loading pre-trained model..."):
            st.session_state.predictor.load_models()
    except FileNotFoundError:
        st.error("Pre-trained model not found. Please run train_model.py first.")
        st.stop()

# Title and description
st.title("📝 Text Personality Analyzer")
st.markdown("""
This app analyzes text content to predict Big Five personality traits:
- **Openness**: Appreciation for art, emotion, adventure, unusual ideas, curiosity, and variety of experience
- **Conscientiousness**: Tendency to be organized and dependable, show self-discipline, act dutifully, aim for achievement
- **Extraversion**: Energy, positive emotions, assertiveness, sociability and the tendency to seek stimulation in the company of others
- **Agreeableness**: Tendency to be compassionate and cooperative rather than suspicious and antagonistic towards others
- **Neuroticism**: Tendency to experience unpleasant emotions easily, such as anger, anxiety, depression, and vulnerability
""")

def analyze_single_text(text_input):
    """Analyze a single text and display results."""
    with st.spinner("Analyzing text..."):
        try:
            # Get predictions
            predictions = st.session_state.predictor.predict(text_input)
            
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            # Display radar chart
            with col1:
                st.subheader("Personality Trait Radar")
                radar_fig = create_trait_radar_chart(list(predictions.values()))
                st.plotly_chart(radar_fig, use_container_width=True)
            
            # Display bar chart
            with col2:
                st.subheader("Trait Scores")
                bar_fig = create_trait_bar_chart(list(predictions.values()))
                st.plotly_chart(bar_fig, use_container_width=True)
            
            # Text Analysis
            st.subheader("Text Analysis")
            
            # Get basic text statistics
            stats = analyze_text_statistics(text_input)
            
            # Get text-specific metrics
            text_metrics = analyze_email_metrics(text_input)
            
            # Create columns for metrics
            metric_cols = st.columns(5)
            
            with metric_cols[0]:
                st.metric("Word Count", stats['word_count'])
            with metric_cols[1]:
                st.metric("Average Sentence Length", f"{stats['avg_sentence_length']:.1f}")
            with metric_cols[2]:
                st.metric("Formality Score", f"{text_metrics['formality_score']:.2f}")
            with metric_cols[3]:
                st.metric("Sentiment Score", f"{text_metrics['sentiment_score']:.2f}")
            with metric_cols[4]:
                st.metric("Urgency Level", text_metrics['response_urgency'])
            
            # Word Cloud
            st.subheader("Key Terms Visualization")
            wordcloud_fig = generate_wordcloud(text_input)
            st.pyplot(wordcloud_fig)
            
            # Feature importance
            st.subheader("Feature Importance Analysis")
            trait_tabs = st.tabs(list(predictions.keys()))
            
            for trait, tab in zip(predictions.keys(), trait_tabs):
                with tab:
                    importance_scores = st.session_state.predictor.get_feature_importance(trait)
                    feature_names = [f"Feature_{i}" for i in range(len(importance_scores))]
                    importance_fig = plot_feature_importance(feature_names, importance_scores)
                    st.plotly_chart(importance_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error analyzing text: {str(e)}")

def analyze_multiple_texts(df):
    """Analyze multiple texts and display aggregate results."""
    with st.spinner("Analyzing texts..."):
        try:
            # Get predictions for all texts
            all_predictions = []
            for text in df['text_content']:
                if pd.isna(text):
                    continue
                predictions = st.session_state.predictor.predict(str(text))
                all_predictions.append(predictions)
            
            # Convert to DataFrame
            predictions_df = pd.DataFrame(all_predictions)
            
            # Display aggregate statistics
            st.subheader("Aggregate Personality Traits")
            
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Display average radar chart
                avg_predictions = predictions_df.mean().to_dict()
                radar_fig = create_trait_radar_chart(list(avg_predictions.values()))
                st.plotly_chart(radar_fig, use_container_width=True)
            
            with col2:
                # Display distribution plot
                st.plotly_chart(
                    create_trait_distribution_plot(predictions_df),
                    use_container_width=True
                )
            
            # Display detailed statistics
            st.subheader("Detailed Statistics")
            st.dataframe(
                predictions_df.describe(),
                use_container_width=True
            )
            
            # Display correlation heatmap
            st.subheader("Trait Correlations")
            correlation_fig = create_correlation_heatmap(predictions_df)
            st.plotly_chart(correlation_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error analyzing texts: {str(e)}")

# Input method selection
input_method = st.radio(
    "Choose input method:",
    ["Single Text", "Multiple Texts (CSV)"]
)

if input_method == "Single Text":
    # Text input for single analysis
    text_input = st.text_area(
        "Enter text content:",
        height=200,
        placeholder="Paste your text content here..."
    )
    
    # Analysis button for single text
    if st.button("Analyze Text"):
        if text_input.strip():
            analyze_single_text(text_input)
        else:
            st.error("Please enter text content to analyze.")

else:
    # File upload for multiple texts
    uploaded_file = st.file_uploader("Upload CSV file with texts", type=['csv'])
    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Check if required column exists
            if 'text_content' not in df.columns:
                st.error("CSV file must contain a 'text_content' column.")
                st.stop()
            
            # Analyze multiple texts
            analyze_multiple_texts(df)
            
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")

def create_trait_distribution_plot(predictions_df):
    """Create a distribution plot for all traits."""
    import plotly.figure_factory as ff
    
    fig = ff.create_distplot(
        [predictions_df[col] for col in predictions_df.columns],
        predictions_df.columns,
        bin_size=0.05
    )
    fig.update_layout(
        title="Trait Score Distributions",
        xaxis_title="Score",
        yaxis_title="Density"
    )
    return fig

def create_correlation_heatmap(predictions_df):
    """Create a correlation heatmap for traits."""
    import plotly.graph_objects as go
    
    corr_matrix = predictions_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title="Trait Correlations",
        width=600,
        height=600
    )
    return fig

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Machine Learning and NLP</p>
</div>
""", unsafe_allow_html=True) 