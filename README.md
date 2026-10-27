# Loan Default Prediction Using Machine Learning

**A comprehensive machine learning project for credit risk analysis, loan default classification, and borrower risk assessment.**

This project uses historical borrower data to predict the likelihood of loan default. It covers exploratory data analysis, feature engineering, imbalanced classification, model comparison, hyperparameter optimization, probability calibration, and decision-threshold tuning.

The final model is a calibrated LightGBM classifier evaluated on a held-out test set, achieving **89.4% accuracy, 77.0% recall for the default class, and a ROC-AUC of 0.921** at the selected decision threshold.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset Description](#dataset-description)
- [Technologies and Libraries](#technologies-and-libraries)
- [Project Workflow](#project-workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Models Evaluated](#models-evaluated)
- [Model Optimization](#model-optimization)
- [Model Evaluation Metrics](#model-evaluation-metrics)
- [Results and Performance](#results-and-performance)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [How to Run the Project](#how-to-run-the-project)
- [Interpretation of Predictions](#interpretation-of-predictions)


---

## Project Overview

Loan default prediction is an important task in financial risk management. Financial institutions need ways to identify applications that may carry a higher risk of default while maintaining fair and informed lending processes.

The objective of this project is to build and evaluate machine learning models that distinguish between borrowers who default and borrowers who do not.

The analysis investigates borrower demographics, income, employment history, property ownership, profession, and geographical information to identify patterns associated with loan default.

Multiple classification algorithms and class-imbalance techniques are compared. The best-performing approach is subsequently optimized, calibrated, and evaluated using an unseen test set.

## Objectives

The main objectives are to:

- Perform exploratory data analysis to understand borrower characteristics and default patterns.
- Examine the distribution of the target variable and identify class imbalance.
- Clean, encode, and transform features for machine learning.
- Engineer additional features representing employment and residential stability.
- Compare resampling and class-weighting approaches.
- Evaluate several classification algorithms using consistent data splits.
- Optimize the selected model through hyperparameter tuning.
- Calibrate predicted probabilities to improve their reliability.
- Tune the classification threshold to improve default detection.
- Evaluate the final model using appropriate classification metrics.
- Identify influential features and discuss their implications for credit risk assessment.

---

## Dataset Description

The dataset was obtained from Kaggle under the title **Loan Prediction Based on Customer Behavior**.

The original dataset is provided as `Training_Data.csv`.

### Dataset summary

| Property | Description |
|---|---|
| Total records | 252,000 |
| Total columns | 13 |
| Input features | 12 |
| Target variable | `Risk_Flag` |
| Missing values | None reported |
| Duplicate records | None reported |
| Non-default cases | 221,004 (87.7%) |
| Default cases | 30,996 (12.3%) |

### Target variable

The target column is `Risk_Flag`.

- `0` — The borrower did not default.
- `1` — The borrower defaulted.

The dataset is imbalanced, with approximately 7.1 non-default cases for every default case. This imbalance is an important consideration when training and evaluating the models.

### Original features

The dataset contains the following columns:

| Feature | Description |
|---|---|
| `ID` | Borrower identifier |
| `Income` | Borrower income |
| `Age` | Borrower age |
| `Experience` | Years of work experience |
| `Married/Single` | Marital status |
| `House_Ownership` | House ownership category |
| `Car_Ownership` | Car ownership status |
| `Profession` | Borrower's profession |
| `CITY` | Borrower's city |
| `STATE` | Borrower's state |
| `CURRENT_JOB_YRS` | Years in the current job |
| `CURRENT_HOUSE_YRS` | Years in the current residence |
| `Risk_Flag` | Loan default target |

The identifier `ID` is included in the source dataset, but should not automatically be treated as a meaningful predictive feature.

---

## Technologies and Libraries

The project uses Python and the following data science and machine learning libraries:

| Library | Purpose |
|---|---|
| Python | Main programming language |
| Jupyter Notebook | Interactive analysis and experimentation |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| SciPy | Statistical analysis |
| Scikit-learn | Preprocessing, model evaluation, calibration, and tuning |
| Imbalanced-learn | Class resampling techniques |
| LightGBM | Gradient-boosting classification |
| XGBoost | Gradient-boosting classification |
| CatBoost | Gradient-boosting classification |

---

## Project Workflow

The project follows a structured machine learning workflow.

1. Load and inspect the dataset.
2. Perform exploratory data analysis.
3. Examine the target distribution and class imbalance.
4. Engineer additional features.
5. Encode categorical variables and prepare numerical features.
6. Split the dataset into training, validation, and testing subsets.
7. Compare class-imbalance strategies.
8. Train and compare classification models.
9. Perform cross-validation.
10. Optimize LightGBM hyperparameters.
11. Calibrate predicted probabilities.
12. Tune the classification threshold.
13. Evaluate the final model on the held-out test set.
14. Analyze feature importance and discuss limitations.

### Data splitting

The dataset was divided using a stratified split:

| Subset | Percentage | Records |
|---|---:|---:|
| Training | 70% | 176,400 |
| Validation | 15% | 37,800 |
| Testing | 15% | 37,800 |
| **Total** | **100%** | **252,000** |

Stratification preserves approximately the same target-class proportions across the subsets.

Preprocessing operations that learn from data, including target encoding and resampling, should be fitted using training data only. Validation data is used for model selection and tuning, while the test set is reserved for final evaluation.

---

## Exploratory Data Analysis

Exploratory data analysis was conducted to investigate the distribution of borrower characteristics and their relationships with loan default.

The analysis included:

- Distribution of default and non-default cases.
- Default rates by marital status.
- Applicant counts by car ownership and house ownership.
- Distributions of income, age, experience, and employment and residential tenure.
- Default rates across common professions.
- Correlation analysis of numerical variables.
- Statistical testing of selected categorical relationships.

### Selected findings

**Marital status:** Single applicants had an observed default rate of 12.53%, compared with 10.25% for married applicants.

**Car ownership:** A chi-square test identified a statistically significant association between car ownership and default status (\(\chi^2 = 145.42,\ p < 0.001\)). Statistical significance does not necessarily imply a large practical effect.

**Numerical features:** Age, experience, and current job tenure had weak negative correlations with the target. Income had a very weak linear correlation and was not statistically significant in the reported correlation test.

**Income distribution:** Income appeared relatively uniformly distributed, which the report notes may indicate that the dataset is synthetically generated.

**Profession:** Default rates varied across professions, suggesting that occupational information may contain predictive patterns.

These are associations observed in the dataset and should not be interpreted as causal relationships.

---

## Feature Engineering

Additional features were created to represent borrower stability and earning capacity.

### 1. Job stability ratio

Measures the proportion of work experience spent in the current job.

Conceptually:

`job_stability_ratio = CURRENT_JOB_YRS / Experience`

### 2. House stability ratio

Represents residential tenure relative to work experience.

Conceptually:

`house_stability_ratio = CURRENT_HOUSE_YRS / Experience`

### 3. Income per experience year

Measures income relative to years of experience, with an adjustment to avoid division by zero.

`income_per_exp_year = Income / (Experience + 1)`

### 4. Age categories

The continuous age feature was grouped into five categories:

- Under 25
- 25–35
- 35–45
- 45–55
- 55 and above

### 5. Categorical encoding

Categorical features were encoded according to their cardinality.

- **One-hot encoding:** Marital status, house ownership, car ownership, and age category.
- **Target encoding:** City, state, and profession.

Target encoding was used to represent high-cardinality categorical features more compactly than one-hot encoding.

---

## Handling Class Imbalance

Because only 12.3% of the observations belong to the default class, models trained without appropriate handling of class imbalance may favor the majority class.

The project compares several techniques.

| Technique | Description |
|---|---|
| SMOTE | Generates synthetic minority-class observations |
| SMOTETomek | Combines SMOTE with Tomek Links cleaning |
| ADASYN | Generates synthetic samples with greater emphasis on difficult minority examples |
| Random undersampling | Reduces the majority-class observations |
| Class weighting | Assigns a greater penalty to errors involving the minority class |

For the boosting models, the reported `scale_pos_weight` value was approximately `7.13`, reflecting the class imbalance in the training data.

The experiments compared how these approaches affected the ability to identify actual defaulters.

Resampling was applied to training data only, rather than to the validation or test sets.

---

## Models Evaluated

The project evaluated the following classification algorithms.

### Logistic Regression

Used as a baseline model. It provides a relatively simple approach for comparison with nonlinear algorithms.

### Decision Tree

Captures nonlinear relationships and provides an interpretable tree-based approach.

### Random Forest

Combines multiple decision trees to improve prediction stability. Different class-imbalance strategies were evaluated.

### XGBoost

A gradient-boosting algorithm capable of capturing complex relationships in structured data.

### LightGBM

A gradient-boosting algorithm selected for further optimization because it performed strongly across the validation and cross-validation experiments.

### CatBoost

A gradient-boosting algorithm included to compare another approach for structured data containing categorical information.

### Stacking Ensemble

Combines LightGBM, XGBoost, CatBoost, and Random Forest predictions, using Logistic Regression as the meta-learner.

---

## Model Optimization

### Cross-validation

Five-fold stratified cross-validation was used to assess model performance across multiple folds.

The reported mean cross-validation AUC scores were:

| Model | Mean CV AUC |
|---|---:|
| LightGBM with class weighting | 0.891 |
| XGBoost with class weighting | 0.870 |
| Random Forest with undersampling | 0.864 |
| Random Forest with class weighting | 0.864 |
| Random Forest with SMOTE | 0.820 |
| Decision Tree with SMOTE | 0.699 |
| Logistic Regression with SMOTE | 0.626 |

LightGBM achieved the highest reported mean cross-validation AUC among these candidates.

### Hyperparameter tuning

`RandomizedSearchCV` was used to search for improved LightGBM hyperparameters.

The reported selected parameters were:

| Parameter | Value |
|---|---:|
| `n_estimators` | 400 |
| `num_leaves` | 127 |
| `learning_rate` | 0.1 |
| `subsample` | 1.0 |
| `colsample_bytree` | 0.6 |

The tuned model achieved a reported validation AUC of 0.916.

### Probability calibration

Class weighting can improve the identification of minority-class cases, but the resulting predicted probabilities may not accurately represent observed default frequencies.

Isotonic regression through `CalibratedClassifierCV` was applied to improve probability calibration.

The reported validation results were:

| Metric | Raw tuned LightGBM | Calibrated LightGBM |
|---|---:|---:|
| Accuracy | 87.4% | 90.0% |
| Macro F1 | 0.766 | 0.766 |
| Weighted F1 | 0.886 | 0.899 |
| ROC-AUC | 0.894 | 0.923 |

These values are the validation-stage results reported in the project, not the final held-out test results.

### Decision-threshold tuning

A classification threshold determines when a predicted probability is converted into a positive default prediction.

The project compared several thresholds:

| Threshold strategy | Cutoff | Accuracy | F1-score | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| Default | 0.50 | 0.900 | 0.589 | 0.585 | 0.593 |
| F1-optimal | 0.33 | 0.896 | 0.641 | 0.755 | 0.557 |
| Accuracy-optimal | 0.47 | 0.901 | 0.623 | 0.667 | 0.585 |
| Closest to 90% accuracy | 0.49 | 0.900 | 0.604 | 0.621 | 0.589 |

The selected threshold was 0.33 because it improved the F1-score and recall for the default class in the reported threshold comparison.

A lower threshold identifies more potential defaulters but can also increase false positives.

### Stacking ensemble

The stacking ensemble achieved the following reported validation performance:

| Metric | Score |
|---|---:|
| Accuracy | 89.6% |
| F1-score | 0.623 |
| Macro F1 | 0.781 |
| Weighted F1 | 0.901 |
| ROC-AUC | 0.903 |

The ensemble did not outperform the calibrated LightGBM model on the reported AUC comparison and introduced additional complexity.

---

## Model Evaluation Metrics

The project evaluates performance using several metrics because accuracy alone can be misleading for imbalanced classification.

**Accuracy:** The proportion of all predictions that are correct.

**Precision:** The proportion of predicted defaults that are actual defaults.

**Recall:** The proportion of actual defaults correctly identified by the model.

**F1-score:** The harmonic mean of precision and recall.

**Macro F1:** Calculates F1 separately for each class and gives both classes equal weight.

**Weighted F1:** Averages class-level F1 scores according to the number of observations in each class.

**ROC-AUC:** Measures the model's ability to rank default cases above non-default cases across classification thresholds.

**H-measure:** Provides a classification-performance measure that incorporates assumptions about the relative costs of classification errors.

**Calibration curve:** Compares predicted probabilities with observed event frequencies.

**Lift analysis:** Examines how effectively the model concentrates actual defaulters among borrowers assigned the highest predicted risk.

The analysis also includes confusion matrices, ROC curves, and feature importance.

---

## Results and Performance

### Final held-out test set

The report identifies calibrated LightGBM with class weighting and a decision threshold of 0.33 as the final model.

| Metric | Test result |
|---|---:|
| Accuracy | 89.4% |
| Precision (default class) | 54.9% |
| Recall (default class) | 77.0% |
| F1-score (default class) | 0.641 |
| ROC-AUC | 0.921 |
| Decision threshold | 0.33 |

### Interpretation

- **89.4% accuracy:** Approximately 89.4% of test-set predictions were correct.
- **54.9% precision:** Of the borrowers predicted to default, approximately 54.9% actually belonged to the default class.
- **77.0% recall:** The model identified approximately 77% of actual default cases.
- **0.641 F1-score:** The model achieved a balance between precision and recall for the default class.
- **0.921 ROC-AUC:** The model demonstrated strong discrimination between default and non-default cases.

The test results are close to the reported validation results, providing evidence of generalization to the held-out sample.

The optimized threshold is intended to prioritize identifying more potential defaults. Its suitability in real lending applications would depend on the costs of missed defaults, false alarms, and the lender's risk tolerance.

---

## Key Findings

The analysis identified several patterns associated with default risk.

1. **Profession and geographical features:** Target-encoded profession, city, and state features ranked highly in the final model's feature-importance analysis.

2. **Employment stability:** Age, total work experience, and current job tenure showed weak negative relationships with default in the reported correlation analysis.

3. **Marital status:** Single applicants had a higher observed default rate than married applicants in this dataset.

4. **Income and engineered features:** Income had a weak linear relationship with default but remained important in the nonlinear LightGBM model, particularly alongside engineered features.

5. **Class imbalance:** Class weighting and gradient-boosting models performed well in the reported model comparisons.

6. **Threshold selection:** A threshold of 0.33 increased default-class recall in the reported validation threshold comparison.

7. **Calibration:** Calibrated LightGBM achieved a reported validation AUC of 0.923 and a held-out test AUC of 0.921.

These results describe patterns in the available historical dataset. They do not prove that a feature causes default or that an individual borrower will default.

---

## Repository Structure

The following is a suggested structure for organizing the GitHub repository:

```text
loan-default-prediction-ml/
│
├── README.md
├── loan_default_prediction.ipynb
├── Loan_Default_Prediction_Report.pdf
├── requirements.txt
│
└── data/
    └── Training_Data.csv
```

The `data/` directory is optional if the dataset is not redistributed. If the dataset's license or usage terms do not permit redistribution, download it from the original source and keep the data file out of the public repository.

The notebook and report filenames above are suggested names. Adjust them to match the files actually uploaded to GitHub.

---

## Installation

### Prerequisites

- Python 3.10 or another version compatible with the selected library releases
- Jupyter Notebook or JupyterLab
- Git

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR-USERNAME/loan-default-prediction-ml.git
cd loan-default-prediction-ml
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

If a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

Alternatively, install the main libraries directly:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn imbalanced-learn lightgbm xgboost catboost jupyter
```

For reproducibility, use the package versions tested with the notebook and record them in `requirements.txt`.

---

## How to Run the Project

1. Clone the repository and install the required libraries.
2. Obtain `Training_Data.csv` from the original Kaggle dataset, subject to its terms of use.
3. Place the dataset in the expected local data directory, or update the path in the notebook.
4. Open the notebook:

   ```bash
   jupyter notebook
   ```

5. Select `loan_default_prediction.ipynb` or the notebook's actual filename.
6. Execute the cells in order, beginning with data loading and preprocessing.
7. Review the exploratory visualizations, model comparisons, evaluation results, and feature-importance analysis.

**Important:** The notebook must have its dataset path and any local file references updated before it can run on another computer. The repository structure in this README is a recommendation, not a guarantee that every listed file is already present.

---

## Interpretation of Predictions

The final model estimates default risk using patterns learned from the training data.

At the selected threshold of 0.33:

- A score at or above the threshold is classified as a predicted default.
- A score below the threshold is classified as a predicted non-default.

The threshold is a classification rule, not a guarantee that a borrower has a particular probability of default. Probability interpretation depends on successful calibration and whether the model remains appropriate for the population being assessed.

In a real-world setting, model outputs should be treated as decision-support information and reviewed alongside relevant financial, regulatory, and fairness considerations.






This project demonstrates an end-to-end approach to imbalanced classification, model comparison, probability calibration, and credit risk analysis using Python.

