# Financial Fraud Detection System

## Usage

## Dataset

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Project Overiew

Recognizing fraudulent activity is extremely important for credit card companies because they want to protect their customers and their financial interests. 

The chosen dataset contains transactions made by credit card holders from Europe during September 2013. The data was subject to a PCA transformation beforehand. The features are scaled and the names of the features are not shown due to privacy reasons. Nevertheless, we can still analyze some important aspects of the dataset. The first thing we must do is gather some basic information about the data. Except for the transaction and amount we don't know what the other columns (V1, V2,..., V28) are.

- The mean of all the amounts made is approximately USD 88, which is relatively small.
- There are no "Null" values, so we don't have to work on ways to replace values.
- Most of the transactions were Non-Fraud (99.83%) of the time, while Fraud transactions occurred (017%) of the time in the data frame.


## Exploratory Data Analysis (EDA)
### Data Distribution
### Correlation Analysis
### Class Imbalance Visualization


## Handling Class Imbalance
### Random Undersampling
### Random Oversampling
### SMOTE (Synthetic Minority Oversampling Technique)


## Data Preprocessing
### Missing Values
### Scaling / Normalization
### Train-Test Split


## Model Selection & Training
### Logistic Regression
### Decision Trees & Random Forest
### Gradient Boosting (XGBoost, LightGBM, CatBoost)
### Neural Networks


## Evaluation Metrics
### Confusion Matrix
### Precision, Recall, F1-Score
### ROC Curve & AUC
### PR Curve


## Hyperparameter Tuning
### Grid Search
### Random Search


## Model Comparison


## Results


## Conclusions


## Repository Structure

```
financial-fraud-detection/
├─ README.md                  # Project documentation
├─ requirements.txt           # Necessary Libraries
├─ .gitignore                 # Ignored Files
├─ data/                      # Data
│  ├─ raw/                    # Raw Data 
├─ notebooks/                 # Jupyter notebooks for exploration
├─ experiments/               # Logs, results, and plots
├─ models/                    # Saved model artifacts
├─ tests/                     # Unit and integration tests
└─ docs/                      # Additional documentation
```

