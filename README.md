# Imbalanced Dataset - Capstone Project DSI (#6)



# 0. Problem Statement & Summary
What are some methods that can be used and applied to highly imbalanced datasets (90/10 or worse) while still producing an accurate model?

- The goal of this was to explore various methods of handling imbalanced datasets (a common problem in the realm of data science).
- Data was pulled from the Kaggle Credit Card Fraud dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud/
- Metrics primary looked at were Accuracy and Sensitivity (Due to desiring to reduce the number of False-Negatives i.e. predicting Non-Fraudulent transaction when actually Fraudulent)
- Findings were that using undersampling methods, the SMOTE & ADASYN oversampling algorithms combined with Logistic Regression performed the best on the imbalanced data.
- It is noted that undersampling methods varied in performance by seed and shouldn't be fully trusted


# 1. Files
As listed above, the file used in raw is from Kaggle. It is too large to uploaded on GitHub and so it's not present in the repository. Below are the Jupyter Notebooks used and what they do!

- `credit_eda.ipynb` is where considerable EDA was done on the credit card file from Kaggle. Some insights in the EDA section!
- `credit_oversample_models.ipynb` Models ran on manually created oversamples
- `credit_SMOTE_ADASYN_models.ipynb` Models ran on oversampling datasets created by running SMOTE & ADASYN
- `credit_undersample_models.ipynb` Models ran on 5 different undersampling datasets manually created from the original dataset


# 2. EDA
The Kaggle dataset imported was a ~285,000 transaction file, with `Time`, `Class` (0 - N.Fraud, 1 - Fraud), `V1` - `V28` columns (anonymized data that was taken from the transactions, with ranges from -50 to 50), and `Amount`. The split of the data was roughly around 99.83% being non-fraudulent transactions, and 0.17% fraudulent. Most models trained on this dataset alone would likely never predict the fraudulent transactions yet be considered 99.83% accurate! This isn't ideal.

In exploring the dataset, I found that aside from some minor correlations between `Amount` and several variables, and small correlations between `Class` and the `V##` variables, all of the `V##` variables didn't appear correlated with each other. (A problem that occurs in highly imbalanced datasets) To get a better picture of this, I employed some bootstrapping on the dataset and sampled the minority class, our fraudulent transactions, to make a balanced dataset for visualization. On top of this, I also created several undersampling datasets, with `random_state` values chosen by my classmates. (This is for allowing for exactly replicable results if you choose to run the code!)

Both the oversampling and undersampling results showed much more obvious correlations between the variables! You can see much more obvious positive and negative correlations between the variables, and how they affect the `Class` variable as well. This is visible in the presentation as a heatmap that is much more colorful, but also visible if the code is run in the EDA workbook.


# 3. Results / Conclusions
So we have four types of datasets:
1. Oversample
2. Undersample
3. ADASYN (Oversample method)
4. SMOTE (Oversample method)

On each of those datasets, 5 different models were run through `GridSearchCV` in an attempt to find the best results on the datasets. Those models were:
- Decision Trees
- Extra Trees
- Random Forest
- Logistic Regression (my personal favorite, results aside)
- $k$-NN

The various results can be found in the non-`EDA` notebooks per sampling model, but simplified tables can be found in the presentation slides (linked at the bottom!). What I found was that the Logistic Regression performed the best when it came to being optimized for Sensitivity. While it was technically less accurate overall (due to incorrecty predicting a lot of transactions as fraudulent when they weren't), it allowed for the fewest transactions to get through that were predicted as Non-Fraudulent when they were actually Fraudulent. There's ample room to improve upon that model and test more models as well, but with the time constraints, it worked the best. The various tree models showed high accuracy, but let more of those False-Negatives slip through compared to the Logistic Regression. 


# 4. Looking Ahead / Next Steps




# 5. Additional Resources:

1. Presentation: https://docs.google.com/presentation/d/1vu8jTozKvblCA7SncaJ3OT4yw4tEImS1eL4M3OClI2c/edit?usp=sharing
2. Kaggle Competition Data: https://www.kaggle.com/mlg-ulb/creditcardfraud/
3. SMOTE/ADASYN Resource: https://www.datasciencecentral.com/profiles/blogs/handling-imbalanced-data-sets-in-supervised-learning-using-family

