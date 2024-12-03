# Spam Detection Project

This project implements a spam detection system using a Multinomial Naive Bayes classifier. It preprocesses the dataset by removing unnecessary features and stopwords to optimize the model's efficiency.

## Features
- Preprocessing:
  - Removal of the "Email No." column.
  - Removal of stopwords (e.g., "the", "and", "is").
- Training:
  - Multinomial Naive Bayes model.
- Evaluation:
  - F1-Score.
  - Precision-Recall Curve.
  - Confusion Matrix.

## Dataset
The dataset contains emails labeled as spam or non-spam. Columns represent the count of the most common words in the emails. The dataset is publicly available and downloaded automatically during script execution.

## Requirements
- Python 3.x
- Libraries:
  - `pandas`
  - `nltk`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/francesco-univr/spam-detection-project.git
