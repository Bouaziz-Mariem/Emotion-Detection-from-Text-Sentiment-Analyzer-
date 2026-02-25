import tkinter as tk
from src.gui import EmotionApp


def main():
    root = tk.Tk()
    root.geometry("900x750")
    app = EmotionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
