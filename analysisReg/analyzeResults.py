import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the data
data = pd.read_csv('predictions_out_of_folds.csv')  # Replace with your actual CSV file name

# Round the target and prediction values
data['target_rounded'] = np.clip(np.round(data['target']), 0, 5).astype(int)
data['prediction_rounded'] = np.clip(np.round(data['prediction']), 0, 5).astype(int)

# Create the confusion matrix
cm = confusion_matrix(data['target_rounded'], data['prediction_rounded'])

# Create labels for the confusion matrix
labels = ['0', '1', '2', '3', '4', '5']

# Create a heatmap of the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

plt.title('Confusion Matrix for XGBoost Tornado Damage Predictions')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# Print the confusion matrix
print("Confusion Matrix:")
print("----------------")
print("    Predicted")
print("    0   1   2   3   4   5")
print("    -------------------")
for i, row in enumerate(cm):
    print(f"{labels[i]} | {' '.join(f'{x:3d}' for x in row)}")

# Print accuracy for each class
print("\nAccuracy for each class:")
print("------------------------")
for i, label in enumerate(labels):
    true_positive = cm[i, i]
    total = np.sum(cm[i, :])
    accuracy = true_positive / total if total > 0 else 0
    print(f"Class {label}: {accuracy:.2%}")

# Overall accuracy
overall_accuracy = np.trace(cm) / np.sum(cm)
print(f"\nOverall Accuracy: {overall_accuracy:.2%}")


# Print classification report
from sklearn.metrics import classification_report
print(classification_report(data['target_rounded'], data['prediction_rounded'], target_names=labels))