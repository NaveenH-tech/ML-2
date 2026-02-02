# ML Assignment 2 – Binary Classification (Mental Health / Depression Prediction)

## (a) Problem Statement

Mental health disorders such as depression are influenced by a combination of personal, social, and psychological factors. Early identification of individuals at risk can help in timely intervention and support.  
The objective of this project is to build and compare multiple machine learning classification models to predict whether an individual is likely to experience depression based on survey-based mental health indicators. The task is framed as a **binary classification problem**, where the target variable represents the presence or absence of depression.

---

## (b) Dataset Description

- **Dataset Name:** Mental Health Data (Binary Classification)  
- **Source:** Kaggle  
  https://www.kaggle.com/code/danuherath/mental-health-data-binary-classification  

- **Problem Type:** Binary Classification  
- **Target Variable:** `Depression` (0 = No, 1 = Yes)

### Dataset Size
- **Training Samples:** 2000 rows  
- **Test Samples:** 700 rows (balanced: 50% class 0, 50% class 1)  
- **Number of Features:** 19  

### Preprocessing
The dataset contains both numerical and categorical features. The following preprocessing steps were applied:

- **Numerical features**
  - Missing values imputed using median
  - Standardization using `StandardScaler` (except for Multinomial Naive Bayes)

- **Categorical features**
  - Missing values imputed using most frequent value
  - One-hot encoding using `OneHotEncoder`

A separate preprocessing pipeline without scaling was used for **Multinomial Naive Bayes** to ensure non-negative inputs.

---

## (c) Models Used and Performance Comparison

All models were trained on the training dataset and evaluated on a **balanced test dataset** using the following metrics:
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC (Matthews Correlation Coefficient)

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9257 | 0.9893 | 0.9745 | 0.8743 | 0.9217 | 0.8560 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| kNN | 0.8900 | 0.9815 | 0.9596 | 0.8143 | 0.8810 | 0.7891 |
| Naive Bayes (Gaussian) | 0.6429 | 0.6429 | 0.5833 | 1.0000 | 0.7368 | 0.4082 |
| Naive Bayes (Multinomial) | 0.5429 | 0.8064 | 0.9688 | 0.0886 | 0.1623 | 0.2052 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## (d) Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieved strong overall performance with high precision and AUC, indicating good linear separability of features. |
| Decision Tree | Achieved perfect performance on the test set, indicating excellent fit, though it may indicate overfitting. |
| kNN | Performed well with high precision but slightly lower recall, showing sensitivity to neighborhood structure. |
| Naive Bayes (Gaussian) | High recall but low precision, indicating a tendency to predict the positive class more often. |
| Naive Bayes (Multinomial) | Poor recall despite high precision, making it less suitable for this dataset. |
| Random Forest (Ensemble) | Delivered perfect performance, benefiting from ensemble averaging and reduced variance. |
| XGBoost (Ensemble) | Achieved perfect scores across all metrics, capturing complex feature interactions effectively. |

---

## Notes

- Models are pretrained offline and loaded in the Streamlit application.
- No model training is performed in the UI.
- Test dataset upload and download functionality is provided as per assignment requirements.
