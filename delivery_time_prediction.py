# Linear Regression Project: Parcel Delivery Time Estimation

# Author: Sahil Shete

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

# 2. Load Data (Assume CSV is provided)
df = pd.read_csv("delivery_data.csv")

# 3. Preprocess Data
df['created_at'] = pd.to_datetime(df['created_at'])
df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])
df['delivery_time_minutes'] = (df['actual_delivery_time'] - df['created_at']).dt.total_seconds() / 60

# 4. Handle Missing Values
df = df.fillna(method='ffill')

# 5. Feature Selection
features = ['total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders', 'distance']
X = df[features]
y = df['delivery_time_minutes']

# 6. Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 8. Model Training
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# 9. Recursive Feature Elimination (RFE)
rfe = RFE(lr, n_features_to_select=3)
rfe.fit(X_train_scaled, y_train)
selected_features = X_train.columns[rfe.support_]
print("Selected Features:", selected_features.tolist())

# 10. Prediction
y_pred = lr.predict(X_val_scaled)

# 11. Evaluation
r2 = r2_score(y_val, y_pred)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"R^2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f} minutes")

# 12. Coefficients
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': lr.coef_})
print(coef_df)

# 13. Residual Plot
residuals = y_val - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()

# 14. Save Model (Optional)
# import joblib
# joblib.dump(lr, 'linear_model.pkl')

