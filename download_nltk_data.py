import nltk

# Download all required NLTK datasets
nltk_datasets = [
    'punkt',
    'stopwords', 
    'averaged_perceptron_tagger',
    'vader_lexicon',
    'wordnet',  # Optional but useful
    'omw-1.4'   # Optional for wordnet
]

print("Downloading NLTK datasets...")
for dataset in nltk_datasets:
    try:
        nltk.download(dataset)
        print(f"✅ Downloaded {dataset}")
    except Exception as e:
        print(f"❌ Failed to download {dataset}: {e}")

print("NLTK data download complete!")