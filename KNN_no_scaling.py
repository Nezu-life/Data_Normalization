# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# April 2024

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import pandas as pd
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

# Read the tabular dataset
dataset_path = sys.argv[1]  # File to be used as the input dataset
df = pd.read_csv(dataset_path)

# Split the dataset into features (X) and labels (y)
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]  # Last column as the target variable
y = y.astype(int)

# Initialize 10-fold stratified split
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize variables to store predictions and real values
all_preds = []
all_y_test = []

# Loop through each fold
for train_index, test_index in kf.split(X, y):
    # Split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(X_train, y_train)

    # Store predictions and real values
    preds = model.predict(X_test)
    all_preds.extend(preds)
    all_y_test.extend(y_test)

# Calculate metrics
balanced_acc = balanced_accuracy_score(all_y_test, all_preds)
f1 = f1_score(all_y_test, all_preds)

# Print metrics
print(f"Balanced Accuracy: {balanced_acc:.2f}")
print(f"F1: {f1:.2f}")

# Confusion Matrix Heatmap
conf_mat = confusion_matrix(all_y_test, all_preds)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')

plt.title('No Scaling')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.show()
