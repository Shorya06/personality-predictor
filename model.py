import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os
from data_processing import process_text, create_synthetic_dataset, process_texts_batch, get_feature_names

class PersonalityPredictor:
    def __init__(self):
        self.models = {
            'openness': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'conscientiousness': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'extraversion': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'agreeableness': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'neuroticism': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        }
        self.is_trained = False
        self.feature_names = get_feature_names()

    def train(self, X=None, y=None, use_synthetic=True):
        if use_synthetic or (X is None and y is None):
            print("Using synthetic dataset for training...")
            df = create_synthetic_dataset()
            X = df['text'].values
            y = df[['openness', 'conscientiousness', 'extraversion', 
                   'agreeableness', 'neuroticism']].values
        
        print("Processing text features...")
        X_features = process_texts_batch(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42
        )
        
        print("\nTraining models...")
        for idx, trait in enumerate(self.models.keys()):
            print(f"\nTraining model for {trait}...")
            self.models[trait].fit(X_train, y_train[:, idx])
            
            importances = self.models[trait].feature_importances_
            print(f"\nTop 5 important features for {trait}:")
            # Get indices of top 5 features
            top_indices = np.argsort(importances)[-5:][::-1]
            for i in top_indices:
                feature_name = self.feature_names[i]
                # Format the feature name for better readability
                if feature_name.startswith('word_pattern_'):
                    feature_name = f"Word Pattern {feature_name.split('_')[-1]}"
                elif feature_name.startswith('tfidf_'):
                    feature_name = f"Word Pattern {feature_name.split('_')[-1]}"
                print(f"{feature_name}: {importances[i]:.4f}")
        
        self.is_trained = True
        return X_test, y_test

    def predict(self, text):
        if not self.is_trained:
            raise ValueError("Model needs to be trained first!")
        
        features = process_text(text)
        
        predictions = {}
        for trait, model in self.models.items():
            pred = model.predict([features])[0]
            pred = max(0, min(1, pred))
            predictions[trait] = pred
        
        return predictions

    def save_models(self, directory='models'):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for trait, model in self.models.items():
            model_path = os.path.join(directory, f'{trait}_model.joblib')
            joblib.dump(model, model_path)

    def load_models(self, directory='models'):
        for trait in self.models.keys():
            model_path = os.path.join(directory, f'{trait}_model.joblib')
            if os.path.exists(model_path):
                self.models[trait] = joblib.load(model_path)
            else:
                raise FileNotFoundError(f"No saved model found for {trait}")
        self.is_trained = True

    def get_feature_importance(self, trait):
        if not self.is_trained:
            raise ValueError("Model needs to be trained first!")
        
        return self.models[trait].feature_importances_ 