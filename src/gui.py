import tkinter as tk
from tkinter import scrolledtext

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.preprocessor import preprocess
from src.emotion_scorer import load_lexicon, score_text
from src.visualizer import plot_radar, plot_timeline, get_word_color


class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Analyzer")

        self.lexicon = load_lexicon()

        # --- Input Section ---
        input_label = tk.Label(root, text="Enter text to analyze:", anchor="w")
        input_label.pack(padx=10, pady=(10, 0), fill=tk.X)

        self.input_text = scrolledtext.ScrolledText(root, height=8, wrap=tk.WORD)
        self.input_text.pack(padx=10, pady=(5, 5), fill=tk.X)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.analyze_btn = tk.Button(btn_frame, text="Analyze", command=self.analyze)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # --- Highlighted Output Section ---
        output_label = tk.Label(root, text="Highlighted output:", anchor="w")
        output_label.pack(padx=10, pady=(5, 0), fill=tk.X)

        self.output_text = tk.Text(root, height=6, wrap=tk.WORD, state=tk.DISABLED)
        self.output_text.pack(padx=10, pady=(5, 5), fill=tk.X)

        # --- Scores Label ---
        self.scores_label = tk.Label(root, text="", justify=tk.LEFT, anchor="w", font=("monospace", 10))
        self.scores_label.pack(padx=10, fill=tk.X)

        # --- Charts Frame ---
        self.chart_frame = tk.Frame(root)
        self.chart_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def clear(self):
        """Reset the app to a fresh state."""
        # Clear input
        self.input_text.delete("1.0", tk.END)

        # Clear highlighted output
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)

        # Clear scores
        self.scores_label.config(text="")

        # Clear charts
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        plt.close("all")

    def analyze(self):
        raw = self.input_text.get("1.0", tk.END).strip()
        if not raw:
            return

        # 1. Preprocess
        stemmed, original_tokens, emoticons = preprocess(raw)

        # 2. Score
        result = score_text(stemmed, self.lexicon, emoticons)

        # 3. Build display-friendly word results by pairing original tokens with scores
        display_word_results = self._build_display_results(original_tokens, result["word_results"])

        # 4. Display highlighted text
        self.display_highlighted(display_word_results)

        # 5. Update scores label
        emotions = result["emotions"]
        dominant = result["dominant"]
        scores_str = " | ".join(f"{e}: {s}" for e, s in emotions.items())
        self.scores_label.config(text=f"Dominant: {dominant.upper()} ({emotions[dominant]})\n{scores_str}")

        # 6. Render charts
        self.render_charts(emotions, result["per_sentence"])

    def _build_display_results(self, original_tokens, scored_word_results):
        """Pair original (unstemmed) tokens with scored emotion dicts for display.

        Both lists have the same structure: list of sentences, each a list of tokens.
        The scorer returns (stemmed_token, scores) — we swap in the original token.
        """
        display = []
        for orig_sent, scored_sent in zip(original_tokens, scored_word_results):
            sentence = []
            for orig_word, (_, scores) in zip(orig_sent, scored_sent):
                sentence.append((orig_word, scores))
            display.append(sentence)
        return display

    def display_highlighted(self, word_results):
        """Show words in the output text widget, colored by their dominant emotion."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)

        for sentence_words in word_results:
            for word, emotions in sentence_words:
                color = get_word_color(emotions)
                if color:
                    tag = f"color_{color}"
                    self.output_text.tag_configure(tag, foreground=color)
                    self.output_text.insert(tk.END, word + " ", tag)
                else:
                    self.output_text.insert(tk.END, word + " ")
            self.output_text.insert(tk.END, "\n")

        self.output_text.config(state=tk.DISABLED)

    def render_charts(self, emotions, per_sentence):
        """Embed radar chart and timeline into the Tkinter window."""
        # Clear previous charts and close their figures to free memory
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        plt.close("all")

        # Radar chart
        radar_fig = plot_radar(emotions)
        radar_canvas = FigureCanvasTkAgg(radar_fig, self.chart_frame)
        radar_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        radar_canvas.draw()

        # Timeline (only if multiple sentences)
        timeline_fig = plot_timeline(per_sentence)
        if timeline_fig:
            timeline_canvas = FigureCanvasTkAgg(timeline_fig, self.chart_frame)
            timeline_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            timeline_canvas.draw()
