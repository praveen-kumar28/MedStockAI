import pandas as pd
import numpy as np
import joblib
from pymongo import MongoClient

# Load saved model and encoders
mlp = joblib.load('models/mlp_model.pkl')
scaler = joblib.load('models/scaler.pkl')
le_medicine = joblib.load('models/mlp_le_medicine.pkl')
le_manufacturer = joblib.load('models/mlp_le_manufacturer.pkl')
le_pack_size = joblib.load('models/mlp_le_pack_size.pkl')

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['medical_inventory']
collection = db['order_details1']

def predict_demand():
    medicine_name = input("Enter the medicine name: ")
    target_month_str = input("Enter the target month (e.g., May 2025): ")

    # Convert input month to period
    target_month = pd.to_datetime(target_month_str).to_period('M')
    past_month1 = target_month - 1  # April
    past_month2 = target_month - 2  # March

    # Load all data for the given medicine
    data = pd.DataFrame(list(collection.find({'medicine_name': medicine_name})))
    if data.empty:
        print(f"No data found for medicine: {medicine_name}")
        return

    data['order_date'] = pd.to_datetime(data['order_date'])
    data['month'] = data['order_date'].dt.to_period('M')

    # Group by month
    monthly_data = data.groupby(['medicine_name', 'manufacturer_name', 'pack_size_label', 'month'])['order_quantity'].sum().reset_index()

    # Get past month quantities
    past1_row = monthly_data[monthly_data['month'] == past_month1]
    past2_row = monthly_data[monthly_data['month'] == past_month2]

    if past1_row.empty or past2_row.empty:
        print(f"Not enough data for prediction in {target_month}")
        return

    past1_qty = past1_row['order_quantity'].values[0]
    past2_qty = past2_row['order_quantity'].values[0]

    # Use other details from the most recent record
    manufacturer_name = past1_row['manufacturer_name'].values[0]
    pack_size_label = past1_row['pack_size_label'].values[0]

    # Encode features
    med_enc = le_medicine.transform([medicine_name])[0]
    man_enc = le_manufacturer.transform([manufacturer_name])[0]
    pack_enc = le_pack_size.transform([pack_size_label])[0]

    X_input = pd.DataFrame([[med_enc, man_enc, pack_enc, past1_qty, past2_qty]],
                           columns=['medicine_name_encoded', 'manufacturer_name_encoded', 'pack_size_label_encoded', 'past_month1', 'past_month2'])

    X_scaled = scaler.transform(X_input)
    y_pred_log = mlp.predict(X_scaled)
    y_pred = np.expm1(y_pred_log)

    print(f"Past Month 1 ({past_month1}): {past1_qty}")
    print(f"Past Month 2 ({past_month2}): {past2_qty}")
    print(f"Expected (average): {(past1_qty + past2_qty) / 2}")
    print(f"Predicted Demand for {target_month} for {medicine_name}: {y_pred[0]:.2f} units")

# Run
if __name__ == "__main__":
    predict_demand()
