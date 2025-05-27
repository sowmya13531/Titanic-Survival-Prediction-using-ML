import seaborn as sns
import matplotlib.pyplot as plt
titanic_data = sns.load_dataset('titanic')
numeric_data = titanic_data.select_dtypes(include=['float64', 'int64'])

# Compute correlation matrix
corr = numeric_data.corr()

# Plot heatmap
sns.heatmap(corr, cmap='viridis', annot=True)
plt.show()
