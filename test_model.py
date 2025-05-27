from model import PersonalityPredictor

def main():
    # Initialize and load the model
    predictor = PersonalityPredictor()
    predictor.load_models()
    
    # Sample text
    text = """I absolutely love exploring new ideas and creative projects! Recently, I've been learning to paint 
    and write poetry in my spare time. While I enjoy socializing with friends at art galleries and cultural events, 
    I also make sure to maintain a well-organized schedule and meet all my deadlines. I'm usually the one planning 
    group activities and making sure everyone's needs are considered. Sometimes I worry about whether my artistic 
    work is good enough, but I try to stay focused on personal growth rather than perfectionism. I find great joy 
    in helping others develop their creative potential and often volunteer to mentor aspiring artists."""
    
    # Get predictions
    result = predictor.predict(text)
    
    # Print results
    print("\nPersonality Trait Scores:")
    for trait, score in result.items():
        print(f"{trait.capitalize()}: {score:.2%}")

if __name__ == "__main__":
    main() 