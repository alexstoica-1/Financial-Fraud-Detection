# Financial Fraud Detection System

## Dataset

The chosen dataset contains transactions made by credit card holders from Europe during September 2013. The data was subject to a PCA transformation beforehand. The features are scaled and the names of the features are not shown due to privacy reasons. Except for the transaction and amount there is no information about the other columns {V1, V2,..., V28}.

Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Project Overiew

Recognizing fraudulent activity is extremely important for credit card companies because they want to protect their customers and their financial interests. 

## Exploratory Data Analysis (EDA)


## Handling Class Imbalance
### Random UnderSampling vs OverSampling
### SMOTE (Synthetic Minority Oversampling Technique)


## Data Preprocessing

## Model Selection & Training
### Logistic Regression (LogReg)
### Decision Trees
### K Nearest Neighbors (KNN)
### Support Vector Classifier (SVC)
### Neural Network


## Evaluation Metrics


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
│  ├─ splits/                 # Splitted data
├─ notebooks/                 # Jupyter notebooks
├─ models/                    # Model Artifacts
└─ tests/                     # Integration tests
```

