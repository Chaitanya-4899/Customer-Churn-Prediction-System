import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import randint, uniform

# Load your original dataset
data = pd.read_csv('D:/CI Data Science Projects 2024/Customer Churn Prediction/Dataset/Telco_customer_churn.csv')

# Handle missing values
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Encode categorical features
data = pd.get_dummies(data, drop_first=True)

# Define your selected features and the target
selected_features = [
    'Monthly Charges', 'Tenure Months', 'Zip Code', 'Latitude', 'Longitude',
    'Dependents_Yes', 'Payment Method_Electronic check', 'Paperless Billing_Yes',
    'Partner_Yes', 'Contract_Two year'
]
X = data[selected_features]
y = data['Churn Value']  # Assuming 'Churn' is the target column

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost with Randomized Search for Hyperparameter Tuning
xgb_param_dist = {
    'n_estimators': randint(50, 150),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

random_search_xgb = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=xgb_param_dist,
    n_iter=10,  # Reduced for faster performance
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1  # Verbose output
)

# Fit the Randomized Search model for XGBoost
random_search_xgb.fit(X_train_scaled, y_train)
best_xgb_model = random_search_xgb.best_estimator_

# XGBoost Evaluation
y_pred_xgb = best_xgb_model.predict(X_test_scaled)
print("\nXGBoost with Randomized Search Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

# Save the trained model
joblib.dump(best_xgb_model, 'xgboost_churn_model_retrained.pkl')
