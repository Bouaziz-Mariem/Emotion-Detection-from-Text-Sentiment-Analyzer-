import re

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

STOP_WORDS = set(stopwords.words("english"))
# Keep negation words — they're critical for the scoring engine
STOP_WORDS -= {
    "not", "no", "nor", "never", "neither",
    "nobody", "nothing", "nowhere", "hardly", "barely",
}

# Text emoticon patterns to extract before cleaning
EMOTICON_PATTERNS = [
    ":)", ":-)", ":(", ":-(", ":D", "D:", ">:(", ";)", ":/", "<3",
]


def extract_emoticons(text):
    """Find and remove emoticons from text before cleaning.

    Returns the cleaned text and a list of found emoticons.
    Emoticons must be extracted first — regex cleaning would destroy them.
    """
    found = []
    for pattern in EMOTICON_PATTERNS:
        while pattern in text:
            found.append(pattern)
            text = text.replace(pattern, " ", 1)
    return text, found


def clean_text(text):
    """Lowercase and strip non-alpha characters, keeping apostrophes and spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z\s']", "", text)
    return text.strip()


def tokenize(text):
    """Split text into sentences, then each sentence into words.

    Returns a list of lists: [[word, word, ...], [word, word, ...]]
    """
    sentences = sent_tokenize(text)
    return [word_tokenize(sentence) for sentence in sentences]


def stem_tokens(tokenized_sentences):
    """Stem every token so it matches the stemmed lexicon keys."""
    return [[stemmer.stem(word) for word in sentence] for sentence in tokenized_sentences]


def remove_stop_words(tokenized_sentences):
    """Remove stop words but keep negation words intact."""
    return [[w for w in sentence if w not in STOP_WORDS] for sentence in tokenized_sentences]


def preprocess(text):
    """Full preprocessing pipeline: extract emoticons → clean → tokenize → remove stops → stem.

    Returns:
        stemmed: list of lists of stemmed tokens (for scoring)
        original_tokens: list of lists of unstemmed tokens (for display/highlighting)
        emoticons: list of emoticon strings found in the text
    """
    # 1. Extract emoticons before cleaning destroys them
    text, emoticons = extract_emoticons(text)

    # 2. Clean the text
    text = clean_text(text)

    # 3. Tokenize into sentences → words
    tokenized = tokenize(text)

    # 4. Remove stop words (keep negation)
    filtered = remove_stop_words(tokenized)

    # 5. Stem for lexicon lookup
    stemmed = stem_tokens(filtered)

    # Also keep the unstemmed filtered tokens for display purposes
    original_tokens = filtered

    return stemmed, original_tokens, emoticons


if __name__ == "__main__":
    test_sentences = [
        "I'm so excited about the trip, but a little nervous too.",
        "She was NOT happy about the terrible news. :(",
        "This is extremely disgusting and I'm very angry!",
        "I was thrilled when I got the job offer :D but terrified about moving.",
    ]

    for text in test_sentences:
        print(f"\nInput: {text}")
        stemmed, original, emoticons = preprocess(text)
        print(f"  Emoticons: {emoticons}")
        print(f"  Original tokens: {original}")
        print(f"  Stemmed tokens:  {stemmed}")
