from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
# Dummy example using Titanic data (you'll need to load your real dataset)
titanic_data = pd.read_csv("train.csv")  # replace with your actual dataset

# Preprocessing
numeric_features = ['Age', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])
# Create full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
# Separate X and y
X = titanic_data[numeric_features]
y = titanic_data["Survived"]  # or your actual target column
# Fit and transform (only works if 'classifier' is not at the end!)
X_transformed = pipeline.named_steps['preprocessor'].fit_transform(X)
