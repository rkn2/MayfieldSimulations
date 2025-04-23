# conda create with python=3.11
# pip install mljar-supervised
# make sure you have brew installed lightgbm not conda

from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
plt.interactive(True)

# Generate a timestamp for the filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load the cleaned dataset
df = pd.read_csv('cleaned_data.csv')

## SET UP X AND Y
# what is my initial Y
# degree_of_damage_u
y = df['degree_of_damage_u']

# what is my X
# Find columns containing "damage" (case-insensitive)
damage_columns = [col for col in df.columns if 'damage' in col.lower()]
# Drop the identified columns
df = df.drop(columns=damage_columns)

# Find columns containing "exist" (case-insensitive)
exist_columns = [col for col in df.columns if 'status_u' in col.lower() or 'exist' in col.lower() or 'demolish' in col.lower()]
# Drop the identified columns
df = df.drop(columns=exist_columns)

# Save the modified DataFrame to a new CSV file
df.to_csv('cleaned_data_no_damage.csv', index=False)

# now load this in as X
X = pd.read_csv('cleaned_data_no_damage.csv')

## START MODELING
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42, stratify=y)

y_train.value_counts().plot(kind='bar',color='green')
y_test.value_counts().plot(kind='bar', color='blue')

results_path = f'automl_results_reg_{timestamp}'

automl = AutoML(
    results_path=results_path,
    ml_task='regression',
   #algorithms=["CatBoost", "Xgboost", "LightGBM", "Random Forest", "Linear", "Decision Tree"],
    explain_level= 2,
    hill_climbing_steps=2,
    top_models_to_improve=2,
    golden_features=False, #on / off when needed
    features_selection=False,
    stack_models=False,
    train_ensemble=False,
    mix_encoding=False,
    eval_metric='mae', # MAE is less affected by outliers and skewed distributions than RMSE
    #eval_metric='accuracy',
    validation_strategy={
        "validation_type": "kfold",
        "k_folds": 5,
        "shuffle": True,
        "stratify": True,
    }
)
automl.fit(X_train, y_train)

predictions = automl.predict(X_test)

score = automl.score(X_test, y_test)
print(score)

#predict_all