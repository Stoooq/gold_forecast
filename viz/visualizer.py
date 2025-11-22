import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

class Visualizer:
    @staticmethod
    def plot_preds(y_true, y_pred, dates):
        dates = pd.to_datetime(dates)

        plt.figure(figsize=(10,4))
        plt.plot(dates, y_true, label="True", linewidth=1.5)
        plt.plot(dates, y_pred, label="Predictions", linewidth=1.5, alpha=0.8)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.title("Price: real vs predictions")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_loss(history):
        plt.figure(figsize=(8,4))
        plt.plot(history["train_loss"], label="train")
        if "val_loss" in history and history["val_loss"]:
            plt.plot(history["val_loss"], label="val")
        plt.legend()
        plt.title("Loss")
        plt.tight_layout()
        plt.show()