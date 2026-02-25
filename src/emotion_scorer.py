import json
import os

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICON_PATH = os.path.join(PROJECT_ROOT, "data", "emotion_lexicon.json")

EMOTIONS = ["joy", "anger", "sadness", "fear", "surprise", "disgust"]

# --- Negation ---
# Stemmed forms since the preprocessor stems tokens before they reach the scorer
NEGATION_WORDS = {
    stemmer.stem(w) for w in [
        "not", "no", "never", "neither", "hardly", "barely",
        "don't", "doesn't", "didn't", "won't", "can't",
        "couldn't", "wouldn't", "shouldn't", "isn't",
        "aren't", "wasn't", "weren't",
    ]
}

EMOTION_FLIP = {
    "joy":      "sadness",
    "sadness":  "joy",
    "anger":    "joy",
    "fear":     "joy",
    "surprise": "surprise",  # "not surprised" is still surprise-related
    "disgust":  "joy",
}

# --- Intensifiers ---
# Map stemmed forms to their multiplier
INTENSIFIERS = {
    stemmer.stem(k): v for k, v in {
        "very": 1.5, "extremely": 1.8, "incredibly": 1.7,
        "really": 1.4, "so": 1.3,
        "slightly": 0.5, "barely": 0.4, "somewhat": 0.6,
        "quite": 1.3,
    }.items()
}

# --- Emoji / Emoticon Map ---
EMOJI_MAP = {
    ":)":  {"joy": 0.7},
    ":-)": {"joy": 0.7},
    ":(":  {"sadness": 0.7},
    ":-(": {"sadness": 0.7},
    ":D":  {"joy": 0.9},
    "D:":  {"fear": 0.6},
    ">:(": {"anger": 0.8},
    ";)":  {"joy": 0.5},
    ":/":  {"sadness": 0.4},
    "<3":  {"joy": 0.8},
}


def load_lexicon(path=LEXICON_PATH):
    with open(path) as f:
        return json.load(f)


def score_word(word, lexicon):
    """Return the emotion dict for a word, or empty dict if not found."""
    return lexicon.get(word, {})


def apply_negation(emotion_scores, negated):
    """If the word was negated, remap its emotions via the flip map."""
    if not negated:
        return emotion_scores
    flipped = {}
    for emotion, score in emotion_scores.items():
        target = EMOTION_FLIP.get(emotion, emotion)
        flipped[target] = flipped.get(target, 0) + score * 0.5  # reduce intensity on flip
    return flipped


def apply_intensifier(emotion_scores, multiplier):
    """Scale all emotion scores by the intensifier multiplier."""
    return {emotion: min(score * multiplier, 1.0) for emotion, score in emotion_scores.items()}


def score_emoticons(emoticons):
    """Score a list of emoticon strings found by the preprocessor."""
    totals = {e: 0 for e in EMOTIONS}
    for emoticon in emoticons:
        for emotion, score in EMOJI_MAP.get(emoticon, {}).items():
            totals[emotion] += score
    return totals


def score_sentence(tokens, lexicon):
    """Score a single sentence (stemmed tokens).

    Returns a dict of emotion totals and a list of (token, {emotion: score}) for highlighting.
    """
    sentence_emotions = {e: 0 for e in EMOTIONS}
    word_results = []
    negated = False
    multiplier = 1.0

    for token in tokens:
        # check for negation
        if token in NEGATION_WORDS:
            negated = True
            word_results.append((token, {}))
            continue

        # check for intensifier
        if token in INTENSIFIERS:
            multiplier = INTENSIFIERS[token]
            word_results.append((token, {}))
            continue

        # score the word
        raw_scores = score_word(token, lexicon)
        if raw_scores:
            adjusted = apply_negation(raw_scores, negated)
            adjusted = apply_intensifier(adjusted, multiplier)
            for emotion, score in adjusted.items():
                sentence_emotions[emotion] += score
            word_results.append((token, adjusted))
        else:
            word_results.append((token, {}))

        # reset modifiers after an emotion word is processed
        negated = False
        multiplier = 1.0

    return sentence_emotions, word_results


def score_text(stemmed_sentences, lexicon, emoticons=None):
    """Score the full text.

    Args:
        stemmed_sentences: list of lists of stemmed tokens from the preprocessor
        lexicon: the emotion lexicon dict
        emoticons: optional list of emoticon strings found by the preprocessor

    Returns a dict with:
        emotions:      normalized 0.0–1.0 scores per emotion
        dominant:       the highest-scoring emotion
        per_sentence:  list of raw score dicts per sentence
        word_results:  list of lists of (token, {emotion: score})
    """
    all_emotions = {e: 0 for e in EMOTIONS}
    sentence_scores = []
    all_word_results = []

    for tokens in stemmed_sentences:
        sent_emotions, word_results = score_sentence(tokens, lexicon)
        sentence_scores.append(sent_emotions)
        all_word_results.append(word_results)
        for emotion, score in sent_emotions.items():
            all_emotions[emotion] += score

    # add emoticon scores to the totals
    if emoticons:
        emoji_scores = score_emoticons(emoticons)
        for emotion, score in emoji_scores.items():
            all_emotions[emotion] += score

    # normalize to 0.0–1.0 range
    max_score = max(all_emotions.values()) if max(all_emotions.values()) > 0 else 1
    normalized = {e: round(s / max_score, 2) for e, s in all_emotions.items()}

    dominant = max(normalized, key=normalized.get)

    return {
        "emotions": normalized,
        "dominant": dominant,
        "per_sentence": sentence_scores,
        "word_results": all_word_results,
    }


if __name__ == "__main__":
    from src.preprocessor import preprocess

    lexicon = load_lexicon()

    test_texts = [
        "I'm so excited about the trip, but a little nervous too.",
        "She was NOT happy about the terrible news. :(",
        "This is extremely disgusting and I'm very angry!",
        "I was thrilled when I got the job offer, but terrified about moving to a new city.",
    ]

    for text in test_texts:
        print(f"\nInput: {text}")
        stemmed, original, emoticons = preprocess(text)
        result = score_text(stemmed, lexicon, emoticons)
        print(f"  Dominant: {result['dominant']}")
        print(f"  Scores:   {result['emotions']}")
        print(f"  Words:    ", end="")
        for sent_words in result["word_results"]:
            for word, scores in sent_words:
                if scores:
                    top = max(scores, key=scores.get)
                    print(f"{word}({top}:{scores[top]:.2f}) ", end="")
        print()
