import json

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd


def plot_confusion_matrix(model, X_test, y_test):
    _ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
    plt.savefig("confusion_matrix.png")


def save_metrics(metrics):
    with open("metrics.json", "w") as fp:
        json.dump(metrics, fp)

def save_predictions(y_test, y_pred):
    df = pd.DataFrame({'true_label': y_test, 'predicted_label': y_pred})
    df.to_csv("predictions.csv")

def save_roc_curve(y_test, y_pred_proba):
    # Calcualte ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    # Store roc curve data
    cdf = pd.DataFrame(np.column_stack([fpr, tpr]), columns=["fpr", "tpr"]).astype(
        float
    )
    cdf.to_csv("roc_curve.csv", index=None)
