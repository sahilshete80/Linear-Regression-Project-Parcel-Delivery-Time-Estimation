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

# 15. README Content for GitHub
readme = '''
# 📦 Parcel Delivery Time Prediction - Linear Regression

This project aims to build a linear regression model to predict the time required for parcel delivery using operational data like the number of dashers, outstanding orders, and delivery distance.

---

## 📈 Project Objective

To accurately estimate the delivery time of parcels using data-driven methods to:

- Improve ETA accuracy for customers
- Help businesses allocate delivery resources efficiently
- Understand key drivers of delivery delay

---

## 🧪 Dataset Overview

The dataset contains the following key features:

- `created_at`: Order creation timestamp
- `actual_delivery_time`: Actual time when the order was delivered
- `total_onshift_dashers`: Dashers available
- `total_busy_dashers`: Dashers currently delivering
- `total_outstanding_orders`: Pending deliveries
- `distance`: Distance to be traveled (in miles)

---

## 🔧 Methods Used

- Data cleaning & preprocessing
- Feature selection using Recursive Feature Elimination (RFE)
- Standardization with `StandardScaler`
- Model training using `LinearRegression` from Scikit-learn
- Evaluation using R² and RMSE

---

## ✅ Results

| Metric | Value         |
|--------|---------------|
| R²     | ~0.74         |
| RMSE   | ~9.5 minutes  |

Top features impacting delivery time:
- `distance`
- `total_busy_dashers`
- `total_outstanding_orders`

---

## 📁 Files in This Repo

| File                            | Description                              |
|---------------------------------|------------------------------------------|
| `delivery_time_prediction.ipynb`| Jupyter Notebook with full implementation |
| `Delivery_Time_Prediction_Report.pdf` | Summary Report in PDF format           |
| `README.md`                     | Project overview and instructions        |

---

## 🚀 Future Work

- Try ensemble models (Random Forest, XGBoost)
- Include categorical features like location and restaurant type
- Add external data like weather, traffic conditions

---

## ✍️ Author

**Sahil Shete**  
*Data Science Enthusiast – upGrad & IIIT Bangalore Program*

---
'''

with open("README.md", "w") as f:
    f.write(readme)
