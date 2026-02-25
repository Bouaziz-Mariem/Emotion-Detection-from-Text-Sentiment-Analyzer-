import matplotlib.pyplot as plt
import numpy as np

EMOTION_COLORS = {
    "joy":      "#FFD700",  # gold
    "anger":    "#FF4444",  # red
    "sadness":  "#4488FF",  # blue
    "fear":     "#9944CC",  # purple
    "surprise": "#FF8800",  # orange
    "disgust":  "#44AA44",  # green
}


def plot_radar(emotions, title="Emotion Profile"):
    """Create a radar (spider) chart showing emotion intensities as a filled polygon."""
    labels = list(emotions.keys())
    values = list(emotions.values())
    values += values[:1]  # close the polygon

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.25, color="steelblue")
    ax.plot(angles, values, color="steelblue", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20)
    return fig


def plot_timeline(per_sentence_scores):
    """Create a line chart showing how emotions shift across sentences.

    Returns None if there's only one sentence (no timeline to show).
    """
    if len(per_sentence_scores) < 2:
        return None

    emotions = list(per_sentence_scores[0].keys())
    x = list(range(1, len(per_sentence_scores) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    for emotion in emotions:
        color = EMOTION_COLORS.get(emotion, None)
        y = [sent[emotion] for sent in per_sentence_scores]
        ax.plot(x, y, label=emotion, marker="o", color=color)

    ax.set_xlabel("Sentence")
    ax.set_ylabel("Intensity")
    ax.set_title("Emotion Timeline")
    ax.legend(loc="upper right", fontsize="small")
    ax.set_xticks(x)
    return fig


def get_word_color(emotion_scores):
    """Return the hex color of the dominant emotion for a word, or None."""
    if not emotion_scores:
        return None
    dominant = max(emotion_scores, key=emotion_scores.get)
    return EMOTION_COLORS.get(dominant)


if __name__ == "__main__":
    # Test with dummy data
    dummy_emotions = {
        "joy": 0.72, "anger": 0.05, "sadness": 0.12,
        "fear": 0.55, "surprise": 0.30, "disgust": 0.01,
    }

    dummy_sentences = [
        {"joy": 0.9, "anger": 0.0, "sadness": 0.0, "fear": 0.1, "surprise": 0.3, "disgust": 0.0},
        {"joy": 0.2, "anger": 0.0, "sadness": 0.1, "fear": 0.8, "surprise": 0.2, "disgust": 0.0},
        {"joy": 0.5, "anger": 0.1, "sadness": 0.3, "fear": 0.2, "surprise": 0.1, "disgust": 0.0},
    ]

    print("Generating radar chart...")
    radar_fig = plot_radar(dummy_emotions)
    radar_fig.savefig("/tmp/test_radar.png", dpi=100, bbox_inches="tight")
    print("  Saved to /tmp/test_radar.png")

    print("Generating timeline chart...")
    timeline_fig = plot_timeline(dummy_sentences)
    timeline_fig.savefig("/tmp/test_timeline.png", dpi=100, bbox_inches="tight")
    print("  Saved to /tmp/test_timeline.png")

    print("\nColor mapping test:")
    test_cases = [
        {"joy": 0.9},
        {"anger": 0.8, "fear": 0.3},
        {},
    ]
    for scores in test_cases:
        print(f"  {scores} -> {get_word_color(scores)}")

    plt.close("all")
    print("\nDone.")
