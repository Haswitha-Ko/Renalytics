import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Enable inline plotting for Jupyter Notebooks
%matplotlib inline

# Load the dataset
file_path = r'C:\Users\hp\Desktop\ckdproject\kidney_disease.csv'
print(f"Loading data from: {file_path}")
data = pd.read_csv(file_path)

# Print the first few rows to ensure data is loaded correctly
print("First few rows of the dataset:")
print(data.head())

# Replace tab characters with a space in column names
data.columns = [col.replace('\t', ' ') for col in data.columns]

# Add a 'notes' column with example text
example_texts = [
    "Patient has high blood pressure and diabetes.",
    "Healthy individual with no major issues.",
    "Patient is experiencing severe kidney pain and high creatinine levels.",
    "Diabetic patient with controlled blood pressure.",
    "Patient has a family history of kidney disease."
]

# Repeat the example texts to match the length of the dataset
data['notes'] = [example_texts[i % len(example_texts)] for i in range(len(data))]

# Check the first few rows to ensure the 'notes' column is added
print("First few rows of the dataset after adding 'notes' column:")
print(data.head())

# Convert infinite values to NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Handle missing values
data.ffill(inplace=True)  # Forward fill to handle missing values

# Encode categorical data
data = pd.get_dummies(data, drop_first=True)

# Ensure 'classification' column exists and update its name if needed
target_column = 'classification_not CKD'
if target_column not in data.columns:
    raise ValueError(f"The '{target_column}' column is not found in the dataset. Please check the column names.")

# Separate features and target
X = data.drop(columns=[target_column])
y = data[target_column]

# Process text data if 'notes' column exists
if 'notes' in data.columns:
    # Create a transformer pipeline for feature extraction
    transformer = pipeline('feature-extraction', model='distilbert-base-uncased')
    # Apply the transformer to the 'notes' column and extract features
    text_features = X['notes'].apply(lambda x: transformer(x)[0])
    # Convert the extracted features to a DataFrame
    text_features = pd.DataFrame(text_features.tolist())
    # Concatenate the text features with the other features
    X = pd.concat([X.drop(columns=['notes']), text_features], axis=1)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Set a font that supports special characters
plt.rcParams['font.family'] = 'DejaVu Sans'

# Generate and display the heatmap of correlations
plt.figure(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, fmt=".2f", linewidths=0.5)
plt.show()

# Additional Plots

# Count Plot
print("Generating Count Plot...")
plt.figure(figsize=(12, 6))
sns.countplot(x=target_column, data=data)
plt.show()

# Bar Plot
print("Generating Bar Plot...")
plt.figure(figsize=(12, 6))
sns.barplot(x=target_column, y='age', data=data)
plt.show()

# Histogram
print("Generating Histogram...")
plt.figure(figsize=(12, 6))
sns.histplot(data['age'], bins=30, kde=True)
plt.show()
