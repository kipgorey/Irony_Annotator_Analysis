import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Ensure labels are consistent and convert them to binary (1 for "iro", 0 for "not")
    data['label'] = data['label'].str.strip()
    data['label_binary'] = data['label'].map({'iro': 1, 'not': 0})

    # Determine the majority label for each instance grouped by id_original
    data['majority_label'] = data.groupby('id_original')['label_binary'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)

    # Create a new column indicating agreement with the majority label
    data['agreement'] = (data['label_binary'] == data['majority_label']).astype(int)

    # Bucket ages by 5-year intervals if Age column exists
    if 'Age' in data.columns:
        data['Age'] = pd.to_numeric(data['Age'], errors='coerce')  # Ensure Age is numeric
        max_age = int(data['Age'].max()) + 5
        bins = list(range(0, max_age, 5))
        labels = [f"{start}-{start + 4}" for start in bins[:-1]]
        data['Age_bucket'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

    # Drop rows with missing labels or majority labels
    data = data.dropna(subset=['agreement', 'majority_label'])
    return data

def filter_small_subgroups(data, demographic_vars, min_annotators=5):
    for var in demographic_vars:
        annotator_counts = data.groupby(var)['user'].nunique()
        small_subgroups = annotator_counts[annotator_counts < min_annotators].index
        data = data[~data[var].isin(small_subgroups)]
    return data

def logistic_regression_analysis(data, demographic_vars):
    # Prepare features
    encoder = OneHotEncoder()
    X = encoder.fit_transform(data[demographic_vars])
    y = data['agreement']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression with Cross-Validation
    model = LogisticRegressionCV(cv=5, max_iter=1000, penalty='l2', scoring='roc_auc', random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Coefficients
    feature_names = encoder.get_feature_names_out(demographic_vars)
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)

    return model, report, coefficients, roc_auc

def detect_anomalies(data, model, X, y, demographic_vars):
    # Get predictions and probabilities
    y_prob = model.predict_proba(X)[:, 1]
    residuals = np.abs(y - y_prob)

    # Add residuals to the DataFrame
    data['residual'] = residuals

    # Flag high residuals as anomalies
    threshold = residuals.mean() + 2 * residuals.std()  # Example: Mean + 2 Std Dev
    data['is_anomaly'] = data['residual'] > threshold

    # Group anomalies by demographic variables
    anomaly_summary = data.groupby(demographic_vars, observed=True).agg({'is_anomaly': 'sum', 'residual': 'mean'})
    return data, anomaly_summary

def plot_coefficients(coefficients, data, demographic_vars, output_dir='plots'):
    plt.figure(figsize=(12, 8))
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    bar_colors = list(cmap((coefficients['Coefficient'] - coefficients['Coefficient'].min()) / (coefficients['Coefficient'].max() - coefficients['Coefficient'].min())))

    sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette=bar_colors)

    # Annotate with number of unique annotators in each subgroup
    for i, feature in enumerate(coefficients['Feature']):
        var_name = feature.split('_')[0]
        if var_name in demographic_vars:
            category = '_'.join(feature.split('_')[1:])  # Handle non-numeric categories
            annotator_count = data[data[var_name] == category]['user'].nunique() if category in data[var_name].unique() else 0
            plt.text(coefficients['Coefficient'].iloc[i], i, f"{annotator_count}", va='center', fontsize=8)

    plt.title('Logistic Regression Coefficients by Demographic Feature')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
        # Save the plot
    import os
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'logistic_regression_coefficients.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()

# Main workflow
def main():
    file_path = 'csv/EPICorpus.csv'

    # Columns to include in analysis
    demographic_vars = ['Age_bucket', 'Sex', 'Ethnicity simplified', 'Country of birth', 'Country of residence', 'Nationality', 'Student status', 'Employment status']

    # Load and preprocess data
    data = load_data(file_path)
    data = preprocess_data(data)

    # Filter small subgroups by number of unique annotators
    data = filter_small_subgroups(data, demographic_vars)

    # Logistic Regression Analysis
    print("\n=== Logistic Regression Analysis ===")
    model, report, coefficients, roc_auc = logistic_regression_analysis(data, demographic_vars)
    print(f"ROC-AUC Score: {roc_auc}")
    print(report)

    # Detect anomalies
    X = OneHotEncoder().fit_transform(data[demographic_vars])
    y = data['agreement']
    data, anomaly_summary = detect_anomalies(data, model, X, y, demographic_vars)
    print("\n=== Anomaly Summary ===")
    print(anomaly_summary)

    # Plot coefficients
    plot_coefficients(coefficients, data, demographic_vars)

if __name__ == "__main__":
    main()
