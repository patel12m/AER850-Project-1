# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Step 2.1: Data Processing
# Read data from a CSV file and convert it into a DataFrame
data = pd.read_csv('C:/Users/patel/Documents/GitHub/AER850F2023/Project 1 Data.csv')

# Step 2.2: Data Visualization
# Perform statistical analysis and visualize the dataset behavior within each class
# For example, you can create a pairplot
sns.pairplot(data=data, hue='maintenance_step')
plt.title("Data Visualization")
plt.show()

# Step 2.3: Correlation Analysis
# Calculate and visualize the correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Step 2.4: Classification Model Development/Engineering
# Split the dataset into features (X) and target (y)
X = data.drop('maintenance_step', axis=1)
y = data['maintenance_step']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the classifier (Random Forest as an example)
classifier = RandomForestClassifier()

# Define hyperparameters to search over using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
best_classifier = grid_search.best_estimator_

# Step 2.5: Model Performance Analysis
# Evaluate the model on the test set
y_pred = best_classifier.predict(X_test_scaled)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Create and visualize a confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 2.6: Model Evaluation
# Save the trained model in joblib format
joblib.dump(best_classifier, 'maintenance_classifier.joblib')

# Predict maintenance steps for new coordinates
new_coordinates = np.array([[9.375, 3.0625, 1.51], [6.995, 5.125, 0.3875], [0, 3.0625, 1.93], [9.4, 3, 1.8], [9.4, 3, 1.3]])
new_coordinates_scaled = scaler.transform(new_coordinates)
predicted_steps = best_classifier.predict(new_coordinates_scaled)

print("Predicted Maintenance Steps for New Coordinates:")
for i, step in enumerate(predicted_steps):
    print(f"Coordinates {i + 1}: Maintenance Step {step}")
