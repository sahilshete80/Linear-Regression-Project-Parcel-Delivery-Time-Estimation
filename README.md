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
