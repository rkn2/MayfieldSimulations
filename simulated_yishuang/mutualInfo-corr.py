import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv('cleaned_data_sim.csv')

# Separate features and target
X = data.drop(['Difference', 'tornado_year', 'estimated', 'simulated_damage'], axis=1)
y = data['Difference']

# Encode categorical variables
le = LabelEncoder()
for col in X.select_dtypes(include=['object']):
    X[col] = le.fit_transform(X[col])

# Correlation Analysis
correlations = X.corrwith(y)
top_correlations = correlations.abs().sort_values(ascending=False).head(10)

# Multiple runs of Mutual Information
n_runs = 100
mi_scores_multiple = []

for _ in range(n_runs):
    mi_scores = mutual_info_regression(X, y)
    mi_scores_multiple.append(mi_scores)

mi_scores_df = pd.DataFrame(mi_scores_multiple, columns=X.columns)

# Get top 10 features based on median MI score
top_mi_features = mi_scores_df.median().sort_values(ascending=False).head(10).index

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15))

# Correlation plot
sns.barplot(x=top_correlations.index, y=top_correlations.values, ax=ax1)
ax1.set_title('Top 10 Correlated Features')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax1.set_ylabel('Absolute Correlation')

# Mutual Information box plot
sns.boxplot(data=mi_scores_df[top_mi_features], ax=ax2)
ax2.set_title(f'Top 10 Features by Mutual Information (Over {n_runs} Runs)')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
ax2.set_ylabel('Mutual Information Score')

plt.tight_layout()
plt.show()

# Print results
print("Top correlated features:")
print(top_correlations)

print(f"\nTop features by Median Mutual Information (over {n_runs} runs):")
print(mi_scores_df[top_mi_features].median().sort_values(ascending=False))