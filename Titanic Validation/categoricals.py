# Check columns types to identify categoricals
print(strat_train_set.dtypes)
X = strat_train_set.drop(['survived'], axis=1)
y = strat_train_set['survived']
# Convert categorical columns to numeric with one-hot encoding
X_encoded = pd.get_dummies(X)
scaler = StandardScaler()
X_data = scaler.fit_transform(X_encoded)
y_data = y.to_numpy()
