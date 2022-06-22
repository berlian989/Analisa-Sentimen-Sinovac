import nltk
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from nltk import tokenize
from PyQt5.QtWidgets import QApplication, QWidget


class Canvas(FigureCanvas):
    def __init__(self, parent):
        super().__init__()
        self.setParent(parent)
        # matplot
        data_cleanpositif = pd.read_csv(
            'C:\data_cleanpositif.csv', encoding='latin1')
        token_space = tokenize.WhitespaceTokenizer()
        all_words = ' '.join(
            [tweet for tweet in data_cleanpositif['Content'].astype('str')])
        token_phrase = token_space.tokenize(all_words)
        frequency = nltk.FreqDist(token_phrase)
        df_frequency = pd.DataFrame(
            {"Word": list(frequency.keys()), "Frequency": list(frequency.values())})
        df_frequency = df_frequency.nlargest(columns="Frequency", n=5)
        plt.figure(figsize=(6, 4))
        self.ax = sns.barplot(data=df_frequency, x="Word",
                              y="Frequency", palette='deep')
        self.ax.set(ylabel="Count")
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        self.resize(800, 600)


class Appdemo(QWidget):
    def __init__(self):
        super().__init__()
        chart = Canvas(self)


app = QApplication(sys.argv)
demo = Appdemo()
plt.show()
sys.exit(app.exec_())
