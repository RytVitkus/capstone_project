# Imbalanced Dataset - Capstone Project DSI (#6)


# 1. Problem Statement & Summary
What are some methods that can be used and applied to highly imbalanced datasets (90/10 or worse) while still producing an accurate model?

- The goal of this was to explore various methods of handling imbalanced datasets (a common problem in the realm of data science).
- Data was pulled from the Kaggle Credit Card Fraud dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud/
- Metrics primary looked at were Accuracy and Sensitivity (Due to desiring to reduce the number of False-Negatives i.e. predicting Non-Fraudulent transaction when actually Fraudulent)
- Findings were that using undersampling methods, the SMOTE & ADASYN oversampling algorithms combined with Logistic Regression performed the best on the imbalanced data.
- It is noted that undersampling methods varied in performance by seed and shouldn't be fully trusted


# 2. 
