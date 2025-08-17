import json

import pandas as pd
from sklearn.model_selection import train_test_split

from metrics_and_plots import plot_confusion_matrix, save_metrics, save_predictions, save_roc_curve
from model import evaluate_model, train_model
from utils_and_constants import PROCESSED_DATASET, load_data, load_hyperparameters



def main():
    X, y = load_data(PROCESSED_DATASET)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)

    # Load hyperparameters from the JSON file
    hyperparameters = load_hyperparameters("raw_dataset/rfc_best_params.json")
    # Train the model using the training set
    model = train_model(X_train, y_train, hyperparameters)
    
    # Calculate test set metrics
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)

    print("====================Test Set Metrics==================")
    print(json.dumps(metrics, indent=2))
    print("======================================================")

    # Save metrics into json file
    save_metrics(metrics)
    plot_confusion_matrix(model, X_test, y_test)
    save_predictions(y_test, y_pred)
    save_roc_curve(y_test, y_pred_proba)



if __name__ == "__main__":
    main()
