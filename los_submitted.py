# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 1. Data Loading
df = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/Datasets/LOS.xlsx')
target_variable = 'Length_of_Stay'

# 2. Data Preprocessing
# 2.1. Remove irrelevant columns
columns_to_remove = ['Diagnosis', 'US_Performed', 'US_Number', 'Diagnosis_Presumptive']
df = df.drop(columns=columns_to_remove, errors='ignore')

# 2.2. Standardize missing values
df = df.replace(['Missing', 'missing'], np.nan)

# 2.3. Separate numerical and categorical columns
df_numeric = df.select_dtypes(include=np.number).copy()
df_categorical = df.select_dtypes(include=['object', 'string']).copy()

# 2.4. One-hot encoding for categorical features
df_encoded = pd.get_dummies(df_categorical, dummy_na=False)
df_combined = pd.concat([df_numeric, df_encoded], axis=1)

# 2.5. KNN Imputation
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df_combined), columns=df_combined.columns)

# 2.6. Categorize LOS into Short Stay (<4 days) and Long Stay (>=4 days)
los_threshold = 4
df_imputed['Long_Stay_Binary'] = (df_imputed[target_variable] >= los_threshold).astype(int)

# 3. Exploratory Data Analysis
# 3.1. Distribution of LOS (continuous)
plt.figure(figsize=(8, 6))
sns.histplot(df[target_variable], bins=30, kde=True, color='blue')
plt.title('Distribution of Length of Stay (Original Continuous)', fontsize=16)
plt.xlabel('Length of Stay (Days)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

correlation_matrix = df_numeric.corr()

# Select the top 10 correlated features (based on absolute correlation with LOS)
# We need to ensure 'Length_of_Stay' is a column in df_numeric for this step
if target_variable in correlation_matrix.columns:
    # Get absolute correlations with the target variable
    correlations_with_target = correlation_matrix[target_variable].abs().sort_values(ascending=False)

    # Get the names of the top 10 features (including the target itself)
    top_features = correlations_with_target.head(10).index.tolist()

    # Filter the original correlation matrix to include only these top features
    top_correlation_matrix = correlation_matrix.loc[top_features, top_features]

    # Visualize the correlation matrix for the top features
    plt.figure(figsize=(10, 8))
    sns.heatmap(top_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'Correlation Matrix for Top 10 Features related to {target_variable} (Original Numeric Data)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()
else:
    print(f"\n'{target_variable}' not found in the original numeric data for correlation analysis.")

# 3.2. Distribution of LOS categories
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='Long_Stay_Binary', data=df_imputed, palette='viridis')
plt.title('Distribution of Length of Stay Categories', fontsize=16)
plt.xlabel('Length of Stay Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Short Stay (<4 days)', 'Long Stay (≥4 days)'])
total = len(df_imputed['Long_Stay_Binary'])
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 4. Model Development
# 4.1. Prepare features and target
X = df_imputed.drop(columns=[target_variable, 'Long_Stay_Binary'])
y = df_imputed['Long_Stay_Binary']

# 4.2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4.3. Initialize classifiers
models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# 4.4. Train and evaluate models
performance_metrics = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    performance_metrics[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }

    from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 4.5. Perform Cross-Validation
cv_scores = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Using StratifiedKFold due to potential class imbalance

print("\nPerforming Cross-Validation (5-fold Stratified):")
for name, model in models.items():
    print(f"  Evaluating {name}...")
    # Use 'roc_auc' as the scoring metric for classification
    scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    cv_scores[name] = scores.mean()
    print(f"    Mean AUC: {scores.mean():.4f}")

# Summarize Cross-Validation Performance
df_cv_scores = pd.DataFrame.from_dict(cv_scores, orient='index', columns=['Mean Cross-Validation AUC'])
print("\nCross-Validation Performance Summary (Mean AUC):")
print(df_cv_scores.to_markdown(floatfmt=".4f"))

# Add cross-validation results to the performance metrics summary
for name, mean_auc in cv_scores.items():
    if name in performance_metrics:
        performance_metrics[name]['Mean CV AUC'] = mean_auc
    else:
         performance_metrics[name] = {'Mean CV AUC': mean_auc}


# Update the performance metrics dataframe to include CV scores
df_metrics = pd.DataFrame(performance_metrics).T

# Reorder columns to put Mean CV AUC alongside AUC from test set
cols = df_metrics.columns.tolist()
if 'Mean CV AUC' in cols:
    cols.insert(cols.index('AUC') + 1, cols.pop(cols.index('Mean CV AUC')))
    df_metrics = df_metrics[cols]


print("\nUpdated Model Performance Summary (including Test Set and Cross-Validation):")
print(df_metrics.to_markdown(floatfmt=".4f"))

# Save updated metrics to CSV
df_metrics.to_csv('model_performance_metrics_with_cv.csv')

# 6. Visualization
# 6.1. ROC Curve Comparison
plt.figure(figsize=(10, 8))

for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    else:
        print(f"Model {name} does not have predict_proba, skipping ROC curve.")

#ROC Curve
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing') # Dashed diagonal line
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curve Comparison of Models', fontsize=16)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('roc_curve_comparison.png', dpi=300)
plt.show()

# Plot confusion matrices
for name, model in models.items():
    model.fit(X_train, y_train) # Retrain the model to ensure y_pred is current
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, linecolor='black',
                xticklabels=['Predicted Short Stay', 'Predicted Long Stay'],
                yticklabels=['Actual Short Stay', 'Actual Long Stay'])
    plt.title(f'Confusion Matrix for {name}', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png', dpi=300)
    plt.show() # Use plt.show() to display each plot

#visualize the top 10 predictors for each model

def plot_top_predictors(models, X_train, n_top=10):
    """
    Visualizes the top N most important predictors for each model, where feature
    importance can be extracted (e.g., tree-based models).

    Args:
        models (dict): A dictionary where keys are model names and values are fitted model objects.
        X_train (pd.DataFrame): The training feature set used to train the models.
        n_top (int): The number of top predictors to display for each model.
    """
    print(f"\nVisualizing Top {n_top} Predictors:")
    for name, model in models.items():
        feature_importances = None

        # Check if the model has 'coef_' (for linear models like Logistic Regression)
        if hasattr(model, 'coef_'):
            # For Logistic Regression, absolute value of coefficients can indicate importance
            # Ensure X_train.columns matches the order of coefficients
            if model.coef_.ndim > 1: # Handle multi-class if needed, here assuming binary
                 feature_importances = pd.Series(np.abs(model.coef_[0]), index=X_train.columns)
            else:
                 feature_importances = pd.Series(np.abs(model.coef_), index=X_train.columns)
            title = f'Top {n_top} Predictors for {name} (Absolute Coefficients)'
            xlabel = 'Absolute Coefficient Value'

        # Check if the model has 'feature_importances_' (for tree-based models)
        elif hasattr(model, 'feature_importances_'):
            feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
            title = f'Top {n_top} Predictors for {name} (Feature Importance)'
            xlabel = 'Feature Importance'

        if feature_importances is not None:
            # Get top N features
            top_features = feature_importances.nlargest(n_top)

            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
            plt.title(title, fontsize=16)
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.gca().invert_yaxis() # Display the most important feature at the top
            plt.tight_layout() # Adjust layout to prevent labels overlapping
            plt.show()
        else:
            print(f"  Model {name} does not have an easily extractable feature importance attribute (like coef_ or feature_importances_), skipping visualization.")

# Call the function to visualize top predictors
plot_top_predictors(models, X_train)


# 5. Summarize Performance
df_metrics = pd.DataFrame(performance_metrics).T
print("\nModel Performance Summary:")
print(df_metrics.to_markdown(floatfmt=".4f"))

# Save metrics to CSV for reproducibility
df_metrics.to_csv('model_performance_metrics.csv')
