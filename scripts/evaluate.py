# This script is used to load a trained model and evaluate its performance on a test dataset.

import pandas as pd
import joblib
from sklearn.metrics import classification_report

# Load the trained model
def load_model(model_file):
    return joblib.load(model_file)

# Load test data and labels
def load_data(data_file, labels_file):
    data = pd.read_csv(data_file, index_col=0)
    labels = pd.read_csv(labels_file, index_col=0)
    return data, labels

# Evaluate the model's performance
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))

def main():
    # Load the trained model
    model = load_model('models/splicing_event_predictor.pkl')

    # Load test data
    data, labels = load_data('data/processed_rna_seq_data.csv', 'data/processed_splicing_labels.csv')

    # Evaluate the model
    evaluate_model(model, data, labels)

if __name__ == '__main__':
    main()
