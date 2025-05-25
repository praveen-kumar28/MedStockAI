import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pymongo import MongoClient
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Connect to MongoDB and Load Data
client = MongoClient("mongodb://localhost:27017/")
db = client['medical_inventory']
collection = db['order_details1']

data = pd.DataFrame(list(collection.find()))
data['order_date'] = pd.to_datetime(data['order_date'])

# Step 2: Group by month and medicine to calculate total monthly demand
data['month'] = data['order_date'].dt.to_period('M')
monthly_data = data.groupby(['medicine_name', 'manufacturer_name', 'pack_size_label', 'month'])['order_quantity'].sum().reset_index()

# Optional: Add price from original dataset (one price per medicine)
data_price = data.drop_duplicates(subset=['medicine_name', 'manufacturer_name', 'pack_size_label'])[
    ['medicine_name', 'manufacturer_name', 'pack_size_label', 'price']
]
monthly_data = pd.merge(monthly_data, data_price, on=['medicine_name', 'manufacturer_name', 'pack_size_label'], how='left')

# Step 3: Sort and create lag features
monthly_data = monthly_data.sort_values(by=['medicine_name', 'month'])
monthly_data['past_month1'] = monthly_data.groupby('medicine_name')['order_quantity'].shift(1)
monthly_data['past_month2'] = monthly_data.groupby('medicine_name')['order_quantity'].shift(2)

# Step 4: Drop rows with insufficient data
monthly_data.dropna(subset=['past_month1', 'past_month2'], inplace=True)

# Step 5: Create target variable
monthly_data['future_demand_quantity'] = monthly_data[['past_month1', 'past_month2']].mean(axis=1)

# Step 6: Label Encoding
le_medicine = LabelEncoder()
le_manufacturer = LabelEncoder()
le_pack_size = LabelEncoder()

monthly_data['medicine_name_encoded'] = le_medicine.fit_transform(monthly_data['medicine_name'])
monthly_data['manufacturer_name_encoded'] = le_manufacturer.fit_transform(monthly_data['manufacturer_name'])
monthly_data['pack_size_label_encoded'] = le_pack_size.fit_transform(monthly_data['pack_size_label'])

# Step 7: Prepare training data
X = monthly_data[['medicine_name_encoded', 'manufacturer_name_encoded', 'pack_size_label_encoded',
                  'price', 'past_month1', 'past_month2']]
y = monthly_data['future_demand_quantity']

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 10: Evaluate Model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Step 11: Save Model and Encoders
joblib.dump(model, 'models/rf_model.pkl')
joblib.dump(le_medicine, 'models/le_medicine.pkl')
joblib.dump(le_manufacturer, 'models/le_manufacturer.pkl')
joblib.dump(le_pack_size, 'models/le_pack_size.pkl')

print("Random Forest model and encoders saved successfully.")
    

plt.figure(figsize=(12, 6))
# Subplot 1: Scatter plot (Actual vs Predicted)
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Random Forest - Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Subplot 2: Line plot (First 50 samples)
plt.subplot(1, 2, 2)
plt.plot(y_test.values[:50], label='Actual Demand', marker='o', linestyle='-', color='green')
plt.plot(y_pred[:50], label='Predicted Demand', marker='x', linestyle='--', color='red')
plt.title('RF - Actual vs Predicted Demand (First 50 Samples)', fontsize=14)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Demand', fontsize=12)
plt.legend()

plt.tight_layout()
plt.show()

