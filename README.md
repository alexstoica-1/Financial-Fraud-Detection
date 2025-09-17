# Financial Fraud Detection System

## Dataset

The chosen dataset contains transactions made by credit card holders from Europe during **September 2013**. The data was subject to a **PCA transformation** beforehand. The features are scaled and the names of the features are not shown due to privacy reasons. Except for the `Time` and `Amount` fields, there is no semantic information about the other columns (`V1`–`V28`).

- Source: [Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- Rows: 284,807  
- Positive (fraud) class ratio: ~0.172% (severe class imbalance)

## Project Overiew

Recognizing fraudulent activity is critical for card issuers to protect customers and reduce chargebacks. This project builds a production-minded pipeline to:

- Explore data characteristics and leakage risks  
- Address **class imbalance** with resampling (Under/Over/SMOTE)  
- Train a suite of models (LogReg, KNN, SVC, Decision Tree, **PyTorch NN**)  
- Evaluate with **precision/recall**, **PR-AUC**, and cost-aware metrics  
- Package artifacts for downstream use

## Exploratory Data Analysis (EDA)

Key checks performed (see `notebook.ipynb`):

- **Class distribution**: confirmed heavy skew toward non-fraud.
- **Feature scales**: PCA components are already standardized; `Amount` additionally scaled via `StandardScaler`/`RobustScaler` where applicable.
- **Time effects**: quick sanity checks on `Time` distributions and potential temporal leakage.
- **Correlation heatmap** (on PCA features): no direct interpretability, but used to spot redundancies.
- **Outlier visualization** for `Amount` to inform scaling and model robustness.

## Handling Class Imbalance

Severe class imbalance can bias models toward the majority class. We tested multiple strategies:

### Random UnderSampling vs OverSampling
- **UnderSampling**: balances classes by reducing majority samples; faster training but risks discarding signal.
- **OverSampling**: duplicates minority samples; preserves majority signal but can overfit.

### SMOTE (Synthetic Minority Oversampling Technique)
- **SMOTE** generates synthetic minority points in feature space.  
- Applied **only on the training set** after a **stratified split** to prevent leakage.  
- In our experiments, SMOTE notably improved **recall** and **PR-AUC** versus raw data and simple oversampling.

## Data Preprocessing

- **Train/Validation/Test**: **Stratified** splits to preserve the fraud ratio.
- **Scaling**:  
  - PCA features: already scaled by design.  
  - `Amount`: scaled (e.g., StandardScaler) to aid distance-based models and NN stability.
- **Class weights**: for some models we compared resampling with **class_weight='balanced'** (where supported).
- **Pipelines**: Scikit-learn pipelines ensure transformations are **fit on train only**.

## Model Selection & Training

We benchmarked classical models first, then moved to a neural network implemented in **PyTorch** (replacing the earlier Keras approach referenced on Kaggle).

### Logistic Regression (LogReg)
- Baseline linear model with/without `class_weight='balanced'`.
- Fast, strong baseline; serves as a calibration reference.

### Decision Trees
- Simple non-linear baseline; prone to overfitting but useful for error analysis.

### K Nearest Neighbors (KNN)
- Distance-based; benefits from scaling and can struggle with extreme imbalance without resampling.

### Support Vector Classifier (SVC)
- Tested with RBF kernel; evaluated with and without class weights, plus SMOTE variants.

### Neural Network (PyTorch)
- Re-implemented the deep model in **PyTorch** (see `pytorch_model.ipynb`).  
- Typical architecture: fully connected layers with ReLU, dropout, and a **sigmoid** output for binary classification.  
- **Focal loss** or **pos_weight** (for `BCEWithLogitsLoss`) explored to emphasize the minority class.  
- Trained with **undersampling / oversampling / SMOTE** variants to compare stability and precision–recall trade-offs.

> Training details: Adam optimizer, early stopping on validation PR-AUC / F1, class-balanced mini-batches where applicable.

## Evaluation Metrics

Because accuracy is misleading under heavy imbalance, we focus on:

- **Precision (fraud)**: How many predicted frauds are truly fraud  
- **Recall (fraud)**: How many actual frauds we catch  
- **F1 (fraud)**: Harmonic mean of precision & recall  
- **ROC-AUC**: Overall separability (can be optimistic under imbalance)  
- **PR-AUC (Average Precision)**: More informative for rare positives  
- **Confusion Matrix**: For threshold and cost trade-off discussions

We report **per-class** metrics and show **PR curves**. Threshold tuning is guided by desired business trade-offs (e.g., maximizing recall at a minimum precision, or minimizing expected cost with an assumed false-positive/false-negative cost ratio).

## Results

Representative outcomes from the notebooks (exact values may vary by random seed and split):

- **Baseline (no resampling, default thresholds)**  
  - PR-AUC ≈ **0.12**  
  - Very low recall at acceptable precision, indicating the need for imbalance handling.

- **SMOTE + classical models (best variant)**  
  - PR-AUC up to ≈ **0.70**  
  - Example operating point observed: **recall ≈ 0.91**, **precision ≈ 0.06** (high-recall setting useful for triage with human review).  
  - Cross-validation training scores for baselines (illustrative): KNN ~**94%**, SVC ~**94%**, Decision Tree ~**91%** (train/CV accuracy only—reported for completeness; we prioritize PR metrics).

- **PyTorch NN (with resampling / class weighting)**  
  - Competitive with classical models when tuned; benefits from **pos_weight/focal loss** and **threshold tuning**.  
  - Best use case: when optimizing a **cost-sensitive threshold** and combining with rules or a second-stage reviewer.

> **Cost-aware thresholding**: We provide confusion matrices and PR curves to select thresholds based on your organization’s tolerated false-positive rate vs. missed-fraud cost.

## Conclusions

- **Imbalance handling is essential**: SMOTE (applied correctly on train only) consistently improves **recall** and **PR-AUC** over naive baselines.  
- **Classical models remain strong** baselines on this PCA-transformed dataset; **LogReg/SVC/KNN** with proper resampling perform competitively.  
- The **PyTorch** neural network offers flexible loss weighting and can match or exceed classical methods with careful tuning, though benefits may be marginal without richer features.  

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

