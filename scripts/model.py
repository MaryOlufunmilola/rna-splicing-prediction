# This script defines the machine learning model (e.g., Random Forest or neural network) and trains it on the preprocessed data.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import joblib

# Load preprocessed RNA-Seq data and splicing labels
def load_data(data_file, labels_file):
    data = pd.read_csv(data_file, index_col=0)
    labels = pd.read_csv(labels_file, index_col=0)
    return data, labels

# Train Random Forest classifier
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Cross-validation of the model
def cross_validate_model(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean()}")

# Save the trained model
def save_model(model, model_file):
    joblib.dump(model, model_file)

def main():
    # Load data
    data, labels = load_data('data/processed_rna_seq_data.csv', 'data/processed_splicing_labels.csv')

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Cross-validation
    cross_validate_model(model, data, labels)

    # Save the trained model
    save_model(model, 'models/splicing_event_predictor.pkl')

if __name__ == '__main__':
    main()
