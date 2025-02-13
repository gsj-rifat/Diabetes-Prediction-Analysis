# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report

#download the dataset freom https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes/data

# Load the dataset
file_path = 'C:\\Users\\nn\Downloads\\5. Project Data Science blog post\\Data\\Healthcare-Diabetes.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

print("Overall information of dataset:")
print(df.info())

# Check for missing values
print("\nSummary of missing values:")
print(df.isnull().sum())

# Basic statistics of the dataset
print("\nBasic statistics of the dataset:")
print(df.describe())

# Drop the ID column
df = df.drop(columns=['Id'])

# statistical outlier detection
# Calculate Z-scores for all numerical columns
z_scores = stats.zscore(df.select_dtypes(include=[np.number]))

# Convert to DataFrame for easy manipulation
z_scores_df = pd.DataFrame(z_scores, columns=df.select_dtypes(include=[np.number]).columns)

# Define a threshold to identify outliers (commonly used threshold is 3)
threshold = 3

# Identify outliers
outliers = (np.abs(z_scores_df) > threshold).any(axis=1)
outliers_df = df[outliers]

print(f'Number of outliers detected: {outliers.sum()}')
print('Outliers:')
print(outliers_df)

# density plot of individual features
# Extract numerical features
numerical_features = df.select_dtypes(include=[np.number]).columns

# Set the number of rows and columns for the subplots
n_rows = (len(numerical_features) + 1) // 2  # 2 columns

fig, axes = plt.subplots(n_rows, 2, figsize=(12, n_rows * 5))
axes = axes.flatten()  # Flatten the array of axes for easy iteration

for i, feature in enumerate(numerical_features):
    ax = axes[i]
    sns.kdeplot(df[feature], shade=True, ax=ax)
    ax.set_title(f'Density Plot of {feature}', fontsize=10)
    ax.set_xlabel(feature, fontsize=6)
    ax.set_ylabel('Density', fontsize=6)

# Remove any empty subplots
for i in range(len(numerical_features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout(pad=6.0)  # Adjust spacing
# plt.show()

# Extract relevant columns
x_column = 'Age'
outcome_column = 'Outcome'
numerical_features = df.select_dtypes(include=[np.number]).columns.drop([x_column, outcome_column])

# Set the number of rows and columns for the subplots
# n_rows = (len(numerical_features) + 1) // 2  # 2 columns

fig, axes = plt.subplots(n_rows, 2, figsize=(12, n_rows * 5))
axes = axes.flatten()  # Flatten the array of axes for easy iteration

for i, feature in enumerate(numerical_features):
    ax = axes[i]
    sns.scatterplot(data=df, x=x_column, y=feature, hue=outcome_column, palette='coolwarm', ax=ax)
    ax.set_title(f'{x_column} vs {feature} by {outcome_column}', fontsize=12)
    ax.set_xlabel(x_column, fontsize=10)
    ax.set_ylabel(feature, fontsize=10)

# Remove any empty subplots
for i in range(len(numerical_features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout(pad=4.0)  # Adjust spacing
# plt.show()

# Create box plots
fig, axes = plt.subplots(n_rows, 2, figsize=(12, n_rows * 5))
axes = axes.flatten()

for i, feature in enumerate(numerical_features):
    ax = axes[i]
    sns.boxplot(data=df, x=outcome_column, y=feature, ax=ax)
    ax.set_title(f'{outcome_column} vs {feature}', fontsize=12)
    ax.set_xlabel(outcome_column, fontsize=10)
    ax.set_ylabel(feature, fontsize=10)

# Remove any empty subplots
for i in range(len(numerical_features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout(pad=4.0)
# plt.show()

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
# plt.show()


# Select only the specified features from the dataframe
selected_features = ['Pregnancies', 'Glucose', 'BMI', 'Age']
X = df[selected_features]
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Dictionary to store performance metrics
performance_metrics = {}

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC Score': roc_auc_score(y_test, y_pred)
    }
    performance_metrics[model_name] = metrics

    # Print classification report
    print(f'Classification Report for {model_name}:\n')
    print(classification_report(y_test, y_pred))

# Convert performance metrics to DataFrame for easier comparison
metrics_df = pd.DataFrame(performance_metrics).T
print(metrics_df)

# Plot performance metrics
metrics_df.plot(kind='bar', figsize=(14, 8))
plt.title('Model Comparison')
plt.ylabel('Score')
plt.show()
