import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_text_for_traits(traits):
    """Generate text that reflects given personality traits."""
    
    # Templates for different personality traits
    openness_phrases = [
        "I love exploring new ideas and experiences.",
        "Art and creativity are essential parts of life.",
        "I enjoy learning about different cultures and perspectives.",
        "Innovation and original thinking drive me.",
        "I'm always curious about how things work."
    ]
    
    conscientiousness_phrases = [
        "It's important to plan ahead and stay organized.",
        "I always make sure to meet deadlines and commitments.",
        "Details matter, and I pay attention to them.",
        "I believe in being prepared and systematic.",
        "Responsibility and reliability are key values."
    ]
    
    extraversion_phrases = [
        "I enjoy meeting new people and socializing.",
        "Group activities and team projects energize me.",
        "I love being the center of attention.",
        "Social gatherings are where I thrive.",
        "I'm outgoing and love to engage with others."
    ]
    
    agreeableness_phrases = [
        "Helping others brings me joy.",
        "Cooperation is better than competition.",
        "I try to see the best in everyone.",
        "Kindness and compassion are important.",
        "I value harmony in relationships."
    ]
    
    neuroticism_phrases = [
        "I sometimes worry about things.",
        "Changes can be stressful and overwhelming.",
        "I'm sensitive to criticism and feedback.",
        "Uncertainty makes me uncomfortable.",
        "I experience mood swings occasionally."
    ]
    
    # Select phrases based on trait scores
    selected_phrases = []
    if traits['openness'] > 0.6:
        selected_phrases.extend(random.sample(openness_phrases, k=random.randint(1, 2)))
    if traits['conscientiousness'] > 0.6:
        selected_phrases.extend(random.sample(conscientiousness_phrases, k=random.randint(1, 2)))
    if traits['extraversion'] > 0.6:
        selected_phrases.extend(random.sample(extraversion_phrases, k=random.randint(1, 2)))
    if traits['agreeableness'] > 0.6:
        selected_phrases.extend(random.sample(agreeableness_phrases, k=random.randint(1, 2)))
    if traits['neuroticism'] > 0.6:
        selected_phrases.extend(random.sample(neuroticism_phrases, k=random.randint(1, 2)))
    
    # Add some random content
    selected_phrases.append(fake.paragraph())
    
    # Shuffle and combine
    random.shuffle(selected_phrases)
    return " ".join(selected_phrases)

def generate_dataset(n_samples=1000):
    """Generate a synthetic dataset with personality traits and corresponding text."""
    data = []
    
    for _ in range(n_samples):
        # Generate random personality traits
        traits = {
            'openness': np.random.beta(5, 5),  # Beta distribution for more realistic scores
            'conscientiousness': np.random.beta(5, 5),
            'extraversion': np.random.beta(5, 5),
            'agreeableness': np.random.beta(5, 5),
            'neuroticism': np.random.beta(5, 5)
        }
        
        # Generate text based on traits
        text = generate_text_for_traits(traits)
        
        # Add to dataset
        data.append({
            'text': text,
            **traits
        })
    
    return pd.DataFrame(data)

def main():
    print("Generating synthetic dataset...")
    df = generate_dataset()
    
    # Save to CSV
    output_path = 'data/personality_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    
    # Print sample statistics
    print("\nDataset Statistics:")
    print(f"Number of samples: {len(df)}")
    print("\nPersonality Trait Distributions:")
    for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
        print(f"\n{trait.capitalize()}:")
        print(df[trait].describe())

if __name__ == "__main__":
    main() 