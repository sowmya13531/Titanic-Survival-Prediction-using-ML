# Titanic-Survival-Prediction-using-ML

This project applies machine learning to predict whether a passenger survived the Titanic disaster using the classic Titanic dataset.

## Overview

- **Dataset:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
- **Goal:** Binary classification – Survived (1) or Not Survived (0)
- **Model Used:** Random Forest Classifier
- **Accuracy:** ~79%

## Workflow

1. **Data Cleaning & Preprocessing**
   - Handled missing values (Age, Embarked)
   - Dropped irrelevant columns (Name, Ticket, Cabin)
   - Encoded categorical variables (Sex, Embarked)
   - Feature scaling for numerical columns

2. **Feature Engineering**
   - Added FamilySize, IsAlone
   - Combined SibSp and Parch into a single feature

3. **Pipeline & Modeling**
   - Used `ColumnTransformer` and `Pipeline` from `sklearn`
   - Applied `StandardScaler` + `OneHotEncoder`
   - Trained a `RandomForestClassifier`

4. **Evaluation**
   - Train-80
   - Test-20
   - Achieved ~79% accuracy on test data

5. **Prediction**
   - Generated survival predictions (0/1) on test dataset

## How to Run

```bash
# Install required libraries
pip install pandas numpy scikit-learn matplotlib seaborn
```
# Run the notebook in Jupyter or JupyterLab
 Go through the '.csv' files and upload them in jupyter notebook.After follow the package installations then go into the folder 'Titanic Validation' after successful validation go into the Evaluation of accuracy.
 

