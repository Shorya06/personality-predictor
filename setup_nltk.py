import nltk
import sys

def download_nltk_resources():
    """Download required NLTK resources with error handling."""
    resources = [
        'punkt',
        'averaged_perceptron_tagger',
        'vader_lexicon',
        'stopwords',
        'words',
        'wordnet'
    ]
    
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    print("Starting NLTK resource download...")
    download_nltk_resources()
    print("All NLTK resources downloaded successfully!")