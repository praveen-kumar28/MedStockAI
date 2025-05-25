import csv
from datetime import datetime
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
mongo_db = client["medical_inventory"]
collection = mongo_db["stock"]

# Clear existing data (optional)
collection.delete_many({})

max_rows = 100  # Set the limit
row_count = 0

with open('Dataset.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row_count >= max_rows:
            break  # Stop after 100 entries
        try:
            item = {
                'id': int(row['id']),
                'name': row['name'].strip(),
                'price': float(row['price']) if row['price'].strip() else 0.0,
                'is_discontinued': row['is_discontinued'].strip().upper() == 'TRUE',
                'manufacturer_name': row['manufacturer_name'].strip(),
                'type': row['type'].strip(),
                'pack_size_label': row['pack_size_label'].strip(),
                'short_composition1': row['short_composition1'].strip(),
                'short_composition2': row['short_composition2'].strip(),
                'stock': int(row['stock']) if row['stock'].strip() else 20,
                'availability': row['availability'].strip(),
                'manufacture_date': datetime.strptime(row['manufacture_date'], '%Y-%m-%d'),
                'expiry_date': datetime.strptime(row['expiry_date'], '%Y-%m-%d')
            }
            collection.insert_one(item)
            row_count += 1
        except Exception as e:
            print(f"Skipping row due to error: {e}")

print(f"✅ Successfully inserted {row_count} documents into MongoDB.")
