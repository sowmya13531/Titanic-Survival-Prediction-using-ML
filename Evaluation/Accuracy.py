from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
# Assuming your full dataset is in a DataFrame called `titanic`
X = titanic_data.drop('survived', axis=1)
y = titanic_data['survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
# Example: define your feature columns
numeric_features = ['age', 'fare']  # replace with actual numeric columns
categorical_features = ['sex', 'embarked']  # replace with actual categorical columns
# Create preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
# Full pipeline with classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
# Fit on training data
pipeline.fit(X_train, y_train)
# Evaluate on test data
score = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {score:.4f}")
