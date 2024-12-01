import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
data = pd.read_csv('csv/EPICorpus.csv')

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
    data['Age_bucket'] = (data['Age'] // 5) * 5

# Prepare demographic features
# Encode categorical variables and scale numerical variables
features = ['Age_bucket', 'Sex', 'Ethnicity simplified', 'Country of birth', 'Country of residence', 'Nationality', 'Student status', 'Employment status']

# Filter available columns
features = [f for f in features if f in data.columns]

# Encode categorical features
encoded_features = []
for feature in features:
    if data[feature].dtype == 'object':
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature].fillna('Unknown'))
    encoded_features.append(feature)

# Drop rows with missing labels, majority labels, or features containing NaN values
data = data.dropna(subset=['agreement', 'majority_label'] + encoded_features)

# Split data into train and test
X = data[encoded_features]
y = data['agreement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Visualize feature coefficients
coefficients = pd.DataFrame({'Feature': encoded_features, 'Coefficient': model.coef_[0]})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(coefficients['Feature'], coefficients['Coefficient'], alpha=0.7)
plt.title('Logistic Regression Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')


os.makedirs('plots', exist_ok=True)
plot_path = os.path.join('plots', 'logistic_regression_coefficients_epicorpus.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
plt.show()

