import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import ConfusionMatrixDisplay

import pickle
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("dataset.csv")  

# Replace missing values
df = df.replace({'yes': 1, 'no': 0, '?': 'Others', 'others': 'Others'})

# Drop rows with invalid results
df = df[df['result'] > -5]

# Create age group column
def convertAge(age):
    if age < 4:
        return 'Toddler'
    elif age < 12:
        return 'Kid'
    elif age < 18:
        return 'Teenager'
    elif age < 40:
        return 'Young'
    else:
        return 'Senior'

df['ageGroup'] = df['age'].apply(convertAge)

# Add engineered features
def add_feature(data):
    data['sum_score'] = 0
    for col in data.loc[:, 'A1_Score':'A10_Score'].columns:
        data['sum_score'] += data[col]
    data['ind'] = data['austim'] + data['used_app_before'] + data['jaundice']
    return data

df = add_feature(df)

# Log transform age
df['age'] = df['age'].apply(lambda x: np.log(x))

# Label encode object columns
def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    return data

df = encode_labels(df)

# Correlation heatmap (optional for visualization)
# sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
# plt.show()

# Feature selection
removal = ['ID', 'age_desc', 'used_app_before', 'austim']
X = df.drop(removal + ['Class/ASD'], axis=1)
y = df['Class/ASD']

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)

# Handle class imbalance
ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Normalize features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_val_scaled = scaler.transform(X_val)

# Train multiple models
models = [LogisticRegression(), XGBClassifier(use_label_encoder=False, eval_metric='logloss'), SVC(kernel='rbf')]

best_model = None
best_score = 0

print("🔎 Model Evaluation:\n")

for model in models:
    model.fit(X_resampled, y_resampled)
    val_preds = model.predict(X_val_scaled)
    val_score = metrics.roc_auc_score(y_val, val_preds)

    print(f'{model.__class__.__name__}:')
    print('  Training AUC:', metrics.roc_auc_score(y_resampled, model.predict(X_resampled)))
    print('  Validation AUC:', val_score)
    print()

    if val_score > best_score:
        best_score = val_score
        best_model = model

# ✅ Save best model
with open("autism_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# ✅ Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"✅ Best model saved as 'autism_model.pkl' ({best_model.__class__.__name__})")
print("✅ Scaler saved as 'scaler.pkl'")
