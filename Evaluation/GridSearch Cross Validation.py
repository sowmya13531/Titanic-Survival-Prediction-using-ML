prod_clf = RandomForestClassifier()
param_grid = [
    {
        "n_estimators": [10, 100, 200, 500],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 3, 4],
    }
]
grid_search = GridSearchCV(prod_clf, param_grid=param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(X_data, y_data)
