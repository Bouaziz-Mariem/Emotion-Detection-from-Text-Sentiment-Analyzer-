import json
import os

from nltk.corpus import wordnet
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# Resolve paths relative to the project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_PATH = os.path.join(PROJECT_ROOT, "data", "seed_words.json")
LEXICON_PATH = os.path.join(PROJECT_ROOT, "data", "emotion_lexicon.json")


def load_seed_words(path=SEED_PATH):
    with open(path) as f:
        return json.load(f)


def get_synonyms(word, max_synsets=3):
    """Pull synonyms from WordNet, limited to the first few synsets to reduce noise."""
    synonyms = set()
    for synset in wordnet.synsets(word)[:max_synsets]:
        for lemma in synset.lemmas():
            clean = lemma.name().replace("_", " ").lower()
            if clean != word:
                synonyms.add(clean)
    return synonyms


def build_lexicon(seed_words):
    """Build an emotion lexicon with stemmed keys.

    Seed words get intensity 0.9, synonyms get 0.6.
    Words appearing under multiple emotions keep all mappings (mixed emotions).
    All keys are stemmed so lookups match the preprocessor output.
    """
    lexicon = {}

    for emotion, seeds in seed_words.items():
        for word in seeds:
            stemmed = stemmer.stem(word)
            # seed word — high confidence
            lexicon.setdefault(stemmed, {})[emotion] = 0.9

            # expand with synonyms — moderate confidence
            for synonym in get_synonyms(word):
                stemmed_syn = stemmer.stem(synonym)
                lexicon.setdefault(stemmed_syn, {}).setdefault(emotion, 0.6)

    return lexicon


def save_lexicon(lexicon, path=LEXICON_PATH):
    with open(path, "w") as f:
        json.dump(lexicon, f, indent=2, sort_keys=True)
    print(f"Lexicon saved to {path}")
    print(f"  Total entries: {len(lexicon)}")
    emotions_count = {}
    for emotions in lexicon.values():
        for emotion in emotions:
            emotions_count[emotion] = emotions_count.get(emotion, 0) + 1
    for emotion, count in sorted(emotions_count.items()):
        print(f"  {emotion}: {count} words")


if __name__ == "__main__":
    seeds = load_seed_words()
    lexicon = build_lexicon(seeds)
    save_lexicon(lexicon)

    # quick spot-check
    print("\n--- Spot Check ---")
    for test_word in ["happi", "angri", "sad", "afraid", "surpris", "disgust"]:
        entry = lexicon.get(test_word, {})
        print(f"  {test_word}: {entry}")
