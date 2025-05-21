import pandas as pd
import numpy as np
from model import PersonalityPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(data_path):
    """Load and prepare the Kaggle dataset."""
    # Load the dataset
    # Expected columns: 'text', 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'
    df = pd.read_csv(data_path)
    
    # Ensure all required columns are present
    required_columns = ['text', 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove any rows with missing values
    df = df.dropna()
    
    # Normalize personality scores to 0-1 range if they aren't already
    trait_columns = required_columns[1:]
    for col in trait_columns:
        if df[col].max() > 1 or df[col].min() < 0:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    return df

def evaluate_model(predictor, X_test, y_test, trait_names):
    """Evaluate model performance and create visualization."""
    # Make predictions
    predictions = []
    for text in X_test:
        pred = predictor.predict(text)
        predictions.append(list(pred.values()))
    predictions = np.array(predictions)
    
    # Calculate metrics
    mse = {}
    r2 = {}
    for i, trait in enumerate(trait_names):
        mse[trait] = mean_squared_error(y_test[:, i], predictions[:, i])
        r2[trait] = r2_score(y_test[:, i], predictions[:, i])
    
    # Create evaluation plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: MSE by trait
    plt.subplot(2, 1, 1)
    sns.barplot(x=list(mse.keys()), y=list(mse.values()))
    plt.title('Mean Squared Error by Trait')
    plt.xticks(rotation=45)
    
    # Plot 2: R² by trait
    plt.subplot(2, 1, 2)
    sns.barplot(x=list(r2.keys()), y=list(r2.values()))
    plt.title('R² Score by Trait')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    plt.close()
    
    return mse, r2

def main():
    # Initialize the predictor
    predictor = PersonalityPredictor()
    
    try:
        # Load and prepare the data
        print("Loading and preparing data...")
        df = load_and_prepare_data('data/personality_data.csv')
        
        # Split the data
        X = df['text'].values
        y = df[['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        print("Training model...")
        predictor.train(X_train, y_train, use_synthetic=False)
        
        # Evaluate the model
        print("Evaluating model...")
        trait_names = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        mse, r2 = evaluate_model(predictor, X_test, y_test, trait_names)
        
        # Print evaluation metrics
        print("\nModel Evaluation Metrics:")
        print("\nMean Squared Error:")
        for trait, score in mse.items():
            print(f"{trait}: {score:.4f}")
        
        print("\nR² Score:")
        for trait, score in r2.items():
            print(f"{trait}: {score:.4f}")
        
        # Save the model
        print("\nSaving model...")
        predictor.save_models()
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main() 