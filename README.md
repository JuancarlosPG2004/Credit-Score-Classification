# Credit Score Classification

**Authors:** Juan Carlos Pajares and Jordy Saltos

## 1. Project Overview

This project focuses on the development of a machine learning model for credit score classification. The objective is to automatically assess the creditworthiness of loan applicants and classify them into two categories: **Good** and **Bad**.

The original dataset contains three credit score categories: `Good`, `Standard`, and `Poor`. For this project, `Standard` and `Poor` were grouped into the `Bad` class, transforming the original problem into a binary classification task.

The project covers the complete machine learning pipeline, from data preprocessing and exploratory analysis to feature engineering, feature selection, model training, hyperparameter optimization, model comparison, and robustness analysis.

## 2. Objective

The main objective is to develop a predictive model capable of assessing the credit risk of new loan applicants.

The problem is formulated as a supervised binary classification problem:

* Learning type: Supervised learning
* Task: Binary classification
* Target variable: `Good` / `Bad`
* Main evaluation metric: F1-score

The F1-score was chosen as the main metric because it provides a balance between precision and recall. Both types of errors are relevant in a credit scoring problem, making it important to consider both measures when evaluating the models.

## 3. Dataset

The project uses the Credit Score Classification dataset.

The dataset is not directly included in the repository. It can be downloaded from the links provided in the notebook.

The notebook contains the necessary link to access and download the original dataset before running the analysis.

The dataset contains financial and credit-related information about customers, including variables related to:

* Annual income
* Monthly income
* Bank accounts
* Credit cards
* Loans
* Outstanding debt
* Credit utilization
* Delayed payments
* Credit inquiries
* Investments
* Account balances
* Credit history

The original target variable contains three categories: `Good`, `Standard`, and `Poor`. For this project, the `Standard` and `Poor` categories were combined into a single `Bad` category.

## 4. Data Preprocessing

A significant part of the project consists of preparing the raw data before applying the machine learning models.

The main preprocessing steps include:

* Analysis of missing values.
* Removal of irrelevant variables and customer identification information.
* Conversion of numerical variables to appropriate numeric formats.
* Cleaning and transformation of categorical variables.
* Treatment of missing values using available customer information when possible.
* Removal of remaining incomplete observations when necessary.
* Detection and treatment of extreme values.
* Encoding of categorical variables.
* Preparation of the final dataset for machine learning.

### Feature Engineering

Several additional variables were created to provide more useful information about the financial situation of each customer.

The engineered features include variables related to:

* Total number of accounts.
* Debt per account.
* Debt-to-income ratio.
* Delayed payments relative to the number of accounts.

These features were introduced to provide the models with additional information derived from the original variables.

## 5. Exploratory Data Analysis

Exploratory data analysis was performed to understand the structure of the dataset and investigate the relationship between the explanatory variables and the target.

The analysis includes:

* Analysis of variable distributions.
* Boxplots.
* Correlation matrices.
* Analysis of numerical variables.
* Analysis of categorical variables.
* Relationship between explanatory variables and the target.
* Mutual Information analysis.

Feature selection was also performed to retain informative variables while limiting redundancy between highly correlated features.

## 6. Class Imbalance

The target variable presents an imbalance between the two classes.

To address this issue during the training process, **SMOTE (Synthetic Minority Oversampling Technique)** was used to increase the representation of the minority class.

The resampling procedure was applied to the training data in order to avoid using information from the test set during model training.

## 7. Machine Learning Models

Four different classification approaches were considered and compared.

### Logistic Regression

Logistic Regression was used as a baseline model. It provides a simple linear approach to the classification problem and allows the performance of more complex models to be compared against a basic reference.

### Random Forest

Random Forest is an ensemble method based on multiple decision trees. It can capture nonlinear relationships between variables and combines the predictions of several trees to improve generalization.

### Bagging Classifier

The Bagging Classifier is an ensemble method that trains several estimators on different samples of the training data and combines their predictions.

### Gradient Boosting

Gradient Boosting builds a sequence of decision trees, with each new tree focusing on correcting errors made by the previous trees. This allows the model to capture complex relationships in the data.

## 8. Hyperparameter Optimization

Hyperparameter optimization was performed using **GridSearchCV with 5-fold cross-validation**.

Several hyperparameter configurations were tested for each model. The best configuration was selected according to the evaluation criterion used during the optimization process.

This procedure allows the different models to be compared using a consistent validation methodology.

## 9. Model Evaluation

The models were evaluated using several performance metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Specificity
* False Positive Rate
* ROC-AUC
* Training time
* Prediction time

The F1-score was considered the main metric because it combines precision and recall into a single measure.

## 10. Results

The optimized models were evaluated on the test set.

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   0.7845 |    0.7871 | 0.7800 |   0.7835 |  0.8470 |
| Random Forest       |   0.9176 |    0.9544 | 0.8770 |   0.9141 |  0.9707 |
| Bagging Classifier  |   0.9475 |    0.9588 | 0.9352 |   0.9468 |  0.9845 |
| Gradient Boosting   |   0.9435 |    0.9509 | 0.9352 |   0.9430 |  0.9850 |

The results show a substantial difference between the baseline Logistic Regression model and the ensemble methods.

The Bagging Classifier obtained an F1-score of **0.9468**, an accuracy of **0.9475**, and an ROC-AUC of **0.9845**.

Gradient Boosting obtained an F1-score of **0.9430**, an accuracy of **0.9435**, and the highest ROC-AUC among the tested models, with a value of **0.9850**.

The detailed comparison and analysis of the models are available in the notebook and in the accompanying presentation.

## 11. Robustness Analysis

A bootstrap analysis was performed after the model comparison in order to study the stability of the final model's performance.

The bootstrap procedure was used to estimate the variability of several evaluation metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* Specificity
* False Positive Rate
* Prediction time

This analysis provides additional information about the robustness of the model and how its performance varies across different samples of the test data.

## 12. Repository Contents

The repository contains the following main files:

```text
.
├── Report_notebook.ipynb
├── ML_project.pdf
└── README.md
```

### Report_notebook.ipynb

This is the main notebook containing the complete project.

It includes:

* Dataset loading.
* Data preprocessing.
* Exploratory data analysis.
* Feature engineering.
* Feature selection.
* Class imbalance treatment.
* Model training.
* Hyperparameter optimization.
* Model evaluation.
* Model comparison.
* Bootstrap robustness analysis.

The dataset required to run the notebook is downloaded using the links provided within the notebook.

### ML_project.pdf

This PDF contains the presentation of the project.

It summarizes the main steps of the analysis, the machine learning methodology, the models considered, and the main results obtained from the notebook.

The PDF is intended to provide a concise overview of the project, while the notebook contains the complete implementation and detailed analysis.

## 13. How to Run the Project

To reproduce the analysis:

1. Clone or download this repository.
2. Open `Report_notebook.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab.
3. Follow the dataset download link provided in the notebook.
4. Download the required dataset.
5. Upload or place the dataset where indicated in the notebook.
6. Run the notebook sequentially.

All the preprocessing, feature engineering, model training, optimization, evaluation, and robustness analysis are implemented in the notebook.

## 14. Project Structure

The overall workflow of the project can be summarized as follows:

```text
Dataset
   |
   v
Data Preprocessing
   |
   v
Exploratory Data Analysis
   |
   v
Feature Engineering
   |
   v
Feature Selection
   |
   v
Class Imbalance Treatment
   |
   v
Model Training
   |
   v
Hyperparameter Optimization
   |
   v
Model Evaluation
   |
   v
Model Comparison
   |
   v
Bootstrap Robustness Analysis
```

## 15. Conclusion

This project presents a complete machine learning approach to credit score classification.

The analysis compares a baseline linear model with several ensemble learning methods and evaluates their performance using multiple classification metrics. The project also includes preprocessing, feature engineering, hyperparameter optimization, class imbalance treatment, and bootstrap analysis to obtain a more complete evaluation of the models.

The complete methodology and implementation are available in `Report_notebook.ipynb`, while `ML_project.pdf` provides a concise presentation of the work and its main results.
