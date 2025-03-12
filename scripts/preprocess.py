# This script handles the preprocessing of RNA-Seq data for RNA splicing event prediction, including normalization, feature extraction, and handling missing data.

import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer
from sklearn.preprocessing import StandardScaler

# Load RNA-Seq data and splicing labels
def load_data(rna_seq_file, labels_file):
    rna_seq_data = pd.read_csv(rna_seq_file, index_col=0)  # RNA-Seq data (features)
    labels = pd.read_csv(labels_file, index_col=0)         # Splicing event labels (target)
    return rna_seq_data, labels

# Preprocess RNA-Seq data
def preprocess_data(rna_seq_data):
    # Handle missing data (imputation)
    imputer = IterativeImputer(max_iter=10, random_state=42)
    rna_seq_data_imputed = pd.DataFrame(imputer.fit_transform(rna_seq_data), columns=rna_seq_data.columns, index=rna_seq_data.index)
    
    # Normalize the data (standardization)
    scaler = StandardScaler()
    rna_seq_data_scaled = pd.DataFrame(scaler.fit_transform(rna_seq_data_imputed), columns=rna_seq_data.columns, index=rna_seq_data.index)
    
    return rna_seq_data_scaled

# Save preprocessed data
def save_preprocessed_data(rna_seq_data_scaled, labels, output_data_file, output_labels_file):
    rna_seq_data_scaled.to_csv(output_data_file)
    labels.to_csv(output_labels_file)

def main():
    # Load data
    rna_seq_file = 'data/rna_seq_splicing_data.csv'
    labels_file = 'data/splicing_labels.csv'
    rna_seq_data, labels = load_data(rna_seq_file, labels_file)

    # Preprocess the data
    rna_seq_data_scaled = preprocess_data(rna_seq_data)

    # Save preprocessed data
    save_preprocessed_data(rna_seq_data_scaled, labels, 'data/processed_rna_seq_data.csv', 'data/processed_splicing_labels.csv')

if __name__ == '__main__':
    main()
