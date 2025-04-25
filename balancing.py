import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv('cleaned_data_latlong.csv')

# Set up y
y = df['degree_of_damage_u']

# Set up X
# Find columns containing "damage" (case-insensitive)
damage_columns = [col for col in df.columns if 'damage' in col.lower()]
# Drop the identified columns
df = df.drop(columns=damage_columns)

# Find columns containing "exist", "status_u", "demolish", "failure", or "after" (case-insensitive)
exist_columns = [col for col in df.columns if 'status_u' in col.lower() or 'exist' in col.lower() or 'demolish' in col.lower() or 'failure' in col.lower() or 'after' in col.lower()]
# Drop the identified columns
df = df.drop(columns=exist_columns)

# Set up X
X = df

# Convert y to integer and handle any non-numeric values
y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the preprocessor and transform the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Apply RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train_preprocessed, y_train)

# Train XGBoost on the resampled data
model = xgb.XGBClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# Transform the test data
X_test_preprocessed = preprocessor.transform(X_test)

# Make predictions
y_pred = model.predict(X_test_preprocessed)

# Print classification report
print(classification_report(y_test, y_pred))

# Create and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Print class distribution before and after resampling
print("Class distribution before resampling:")
print(y_train.value_counts(normalize=True))

print("\nClass distribution after resampling:")
print(pd.Series(y_resampled).value_counts(normalize=True))

# Feature importance
feature_importance = model.feature_importances_
feature_names = (numeric_features.tolist() +
                 preprocessor.named_transformers_['cat']
                 .get_feature_names_out(categorical_features).tolist())
feature_importance = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 most important features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.show()
