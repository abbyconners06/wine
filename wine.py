import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler

# Load the wine dataset from a CSV file
wine_data = pd.read_csv('wine_combined_cleaned.csv')

# Separate the features (X) and the target variable (y)
X = wine_data.iloc[:, :-1]  # Select all columns except the last one
y = wine_data.iloc[:, -1]   # Select only the last column

# Scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform Random Oversampling on the entire dataset
oversampler = RandomOverSampler(random_state=42)
X_oversampled, y_oversampled = oversampler.fit_resample(X_scaled, y)

# Create individual classifiers
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
svc = SVC(random_state=42)
knn = KNeighborsClassifier()
gradient_boosting = GradientBoostingClassifier(random_state=42)
logistic_regression = LogisticRegression(random_state=42)

# Create the VotingClassifier ensemble
ensemble_model = VotingClassifier(
    estimators=[
        ('dt', decision_tree),
        ('rf', random_forest),
        ('svc', svc),
        ('knn', knn),
        ('gb', gradient_boosting),
        ('lr', logistic_regression)
    ],
    voting='hard'  # Use majority voting
)

# Perform k-fold cross-validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, test_index in kf.split(X_oversampled):
    X_train, X_test = X_oversampled[train_index], X_oversampled[test_index]
    y_train, y_test = y_oversampled[train_index], y_oversampled[test_index]

    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)

    # Predict the quality of wines in the test set
    y_pred = ensemble_model.predict(X_test)

    # Evaluate the model for this fold
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# Compute the average scores across all folds
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
average_precision = sum(precision_scores) / len(precision_scores)
average_recall = sum(recall_scores) / len(recall_scores)
average_f1 = sum(f1_scores) / len(f1_scores)

print("Average Accuracy:", average_accuracy)
print("Average Precision:", average_precision)
print("Average Recall:", average_recall)
print("Average F1-score:", average_f1)
