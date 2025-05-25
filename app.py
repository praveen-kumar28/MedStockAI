from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, send_file
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import StandardScaler
from bson import ObjectId
import re
import joblib
import csv
import os
import pdfplumber
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import NotFittedError
from fpdf import FPDF 

app = Flask(__name__, template_folder='Pages')

app.config['MONGO_URI'] = 'mongodb://localhost:27017/medical_inventory'
app.config['SECRET_KEY'] = 'secretkey'
mongo = PyMongo(app)

client = MongoClient("mongodb://localhost:27017/")
mongo_db = client["medical_inventory"]
collection = mongo_db["stock"]

# Load Random Forest model and encoders
rf_model = joblib.load('models/rf_model.pkl')
le_medicine_rf = joblib.load('models/le_medicine.pkl')
le_manufacturer_rf = joblib.load('models/le_manufacturer.pkl')
le_pack_size_rf = joblib.load('models/le_pack_size.pkl')

# Load MLP model and encoders (renamed for clarity)
mlp_model = joblib.load('models/mlp_model.pkl')
scaler=joblib.load('models/scaler.pkl')
le_medicine_mlp = joblib.load('models/mlp_le_medicine.pkl')
le_manufacturer_mlp = joblib.load('models/mlp_le_manufacturer.pkl')
le_pack_size_mlp = joblib.load('models/mlp_le_pack_size.pkl')

# Dictionary to track the last email sent for each product
last_email_sent = {}

def load_stock_data():

    client = MongoClient("mongodb://localhost:27017/")
    mongo_db = client["medical_inventory"]
    collection = mongo_db["stock"]
    
    stock_data = list(collection.find({}))
    current_date = datetime.now().date()
    out_of_stock_products = []

    for item in stock_data:
        item['manufacture_date'] = item['manufacture_date'].date()
        item['expiry_date'] = item['expiry_date'].date()

        if item['availability'].lower() == "out of stock":
            product_name = item['name']
            if product_name not in last_email_sent or last_email_sent[product_name] != current_date:
                out_of_stock_products.append(product_name)
                last_email_sent[product_name] = current_date

    if out_of_stock_products:
        send_out_of_stock_products_email(out_of_stock_products)

    return stock_data

def generate_order_bill(order_details):
    """
    Generate an order bill as a PDF file.
    """
    pdf_file = "order_bill.pdf"
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Order Bill", ln=True, align='C')
        pdf.ln(10)
        
        # Calculate total amount
        total_amount = order_details['Price'] * order_details['Ordered Quantity']
        order_details['Total Amount'] = total_amount  # Add total amount to order details

        # Write order details to the PDF
        for key, value in order_details.items():
            pdf.cell(200, 10, txt=f"{key}: {value}", ln=True, align='L')
        pdf.output(pdf_file)
        print(f"Order bill generated: {pdf_file}")
    except Exception as e:
        print(f"Error generating order bill: {e}")
    extract_order_data(pdf_file)
    return pdf_file

def extract_order_data(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = pdf.pages[0].extract_text()

        if not text:
            print("No text found in PDF.")
            return None

        # Parse key-value pairs from text
        order_info = {}
        for line in text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                order_info[key.strip()] = value.strip()

        # Extract relevant fields
        record = {
            'medicine_name': order_info.get('Medicine Name'),
            'manufacturer_name': order_info.get('Manufacturer Name', 'Unknown'),
            'pack_size_label': order_info.get('Pack Size Label'),
            'price': float(order_info.get('Price', 0.0)),
            'order_date': datetime.now().strftime('%Y-%m-%d'),
            'order_quantity': int(order_info.get('Ordered Quantity', 0))
            
        }

        # Insert into MongoDB
        client = MongoClient("mongodb://localhost:27017/")
        db = client["medical_inventory"]
        collection = db["order_details1"]
        collection.insert_one(record)
        print("Order record inserted into MongoDB.")

        return pd.DataFrame([record])  # Return as DataFrame for inspection if needed

    except Exception as e:
        print(f"Error extracting order data: {e}")
        return None


def send_out_of_stock_products_email(products, order_details=None):
    # Set up the email server (assuming you're using Gmail's SMTP)
    sender_email = "praveenhardik1548@gmail.com"
    receiver_email = "praveenhardik2826@gmail.com"
    password = "pcgz vspu qhot proo"
    
    # Initialize variables for cleanup
    csv_filename = "out_of_stock_products.csv"
    order_bill_file = None

    # Create the email content
    subject = "Out of Stock Products Alert"
    body = "The following products are currently out of stock:\n\n" + "\n".join(products)

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Generate a CSV file for the out-of-stock products
    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            fieldnames = ['Product Name']
            csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for product in products:
                if isinstance(product, dict) and 'name' in product:
                    csv_writer.writerow({'Product Name': product['name']})
                else:
                    csv_writer.writerow({'Product Name': product})
        
        with open(csv_filename, "rb") as file:
            attachment = MIMEText(file.read(), 'base64', 'utf-8')
            attachment.add_header('Content-Disposition', f'attachment; filename="{csv_filename}"')
            msg.attach(attachment)
    except Exception as e:
        print(f"Error creating or attaching CSV file: {e}")
        return

    # Generate an order bill if order details are provided
    if order_details:
        order_bill_file = generate_order_bill(order_details)
        try:
            with open(order_bill_file, "rb") as file:
                attachment = MIMEText(file.read(), 'base64', 'utf-8')
                attachment.add_header('Content-Disposition', f'attachment; filename="{order_bill_file}"')
                msg.attach(attachment)
        except Exception as e:
            print(f"Error attaching order bill PDF: {e}")

    # Send the email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Out-of-stock products email sent successfully.")
    except Exception as e:
        print(f"Failed to send out-of-stock products email: {str(e)}")
    finally:
        # Clean up the generated files
        if os.path.exists(csv_filename):
            os.remove(csv_filename)
        if order_bill_file and os.path.exists(order_bill_file):
            os.remove(order_bill_file)

# Function to send daily reports of products about to expire as a CSV file
def send_daily_expiry_report():
    stock_data = load_stock_data()
    about_to_expire_products = []
    current_date = datetime.now().date()
    one_month_later = current_date + timedelta(days=30)

    for item in stock_data:
        if current_date <= item['expiry_date'] <= one_month_later:
            about_to_expire_products.append(item)

    if about_to_expire_products:
        # Create a CSV file for the report
        csv_filename = "daily_expiry_report.csv"
        try:
            with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
                fieldnames = ['id', 'name', 'price', 'is_discontinued', 'manufacturer_name', 'type', 
                              'pack_size_label', 'short_composition1', 'short_composition2', 'stock', 
                              'availability', 'manufacture_date', 'expiry_date']
                csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
                csv_writer.writeheader()
                cleaned_data = [{k: v for k, v in item.items() if k != '_id'} for item in about_to_expire_products]
                csv_writer.writerows(cleaned_data)
        except Exception as e:
            print(f"Error creating CSV file: {e}")
            return

        # Set up the email server (assuming you're using Gmail's SMTP)
        sender_email = "praveenhardik1548@gmail.com"
        receiver_email = "praveenhardik2826@gmail.com"
        password = "pcgz vspu qhot proo"

        subject = "Daily Expiry Report"
        body = "The attached file contains the list of products about to expire within 30 days."

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Attach the CSV file
        try:
            with open(csv_filename, 'rb') as file:
                attachment = MIMEText(file.read(), 'base64', 'utf-8')
                attachment.add_header('Content-Disposition', f'attachment; filename="{csv_filename}"')
                msg.attach(attachment)
        except Exception as e:
            print(f"Error attaching CSV file: {e}")
            return

        # Send the email
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            server.quit()
            print("Daily expiry report email sent successfully.")
        except Exception as e:
            print(f"Failed to send daily expiry report email: {str(e)}")
        finally:
            # Clean up the CSV file after sending
            if os.path.exists(csv_filename):
                os.remove(csv_filename)

# Call the daily expiry report function at startup
send_daily_expiry_report()

# Function to get stock details based on the user input
def get_stock_availability(user_input):
    stock_data = load_stock_data()
    product_name = user_input.strip().lower()  # Normalize user input
    matching_products = []

    print(f"DEBUG: Searching for product: '{product_name}'")  # Debugging log

    for item in stock_data:
        normalized_name = item['name'].strip().lower()  # Normalize product name
        print(f"DEBUG: Comparing with product: '{normalized_name}'")  # Debugging log
        if product_name in normalized_name:  # Partial matching
            matching_products.append(item)

    if len(matching_products) == 1:
        item = matching_products[0]
        stock_quantity = item['stock']
        manufacture_date = item['manufacture_date'].strftime('%d-%m-%Y')  # Format manufacture date
        expiry_date = item['expiry_date'].strftime('%d-%m-%Y')
        threshold = 0  # Use a fixed threshold (you can change this based on your logic)

        # Check if the product is expiring within 2 months
        current_date = datetime.now().date()
        two_months_later = current_date + timedelta(days=60)
        if current_date <= item['expiry_date'] <= two_months_later:
            send_expiry_alert_email(item['name'], expiry_date)

        # If stock is below threshold, trigger an alert
        if stock_quantity <= threshold:
            return f"Alert: Stock of {item['name']} is below threshold. Only {stock_quantity} items left. Manufacture Date: {manufacture_date}, Expiry Date: {expiry_date}"

        # If stock is available, return stock details
        return f"Stock for {item['name']}: {stock_quantity} items available. Manufacture Date: {manufacture_date}, Expiry Date: {expiry_date}"

    elif len(matching_products) > 1:
        product_list = ', '.join([item['name'] for item in matching_products])
        return f"Multiple products found: {product_list}. Please specify more clearly."

    print(f"DEBUG: No matching product found for '{product_name}'")  # Debugging log
    return "Sorry, we couldn't find that product."

# Function to send email alert for products expiring within 2 months
def send_expiry_alert_email(product_name, expiry_date):
    sender_email = "praveenhardik1548@gmail.com"
    receiver_email = "praveenhardik2826@gmail.com"
    password = "pcgz vspu qhot proo"
    
    subject = f"Expiry Alert: {product_name}"
    body = f"The product {product_name} is expiring soon. Expiry Date: {expiry_date}."

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"Expiry alert email sent for {product_name}")
    except Exception as e:
        print(f"Failed to send expiry alert email: {str(e)}")
"""
# Function to send email alert when stock is below threshold
def send_stock_alert_email(product_name, stock_quantity, threshold):
    # Set up the email server (assuming you're using Gmail's SMTP)

    sender_email = "praveenhardik1548@gmail.com"
    receiver_email = "praveenhardik2826@gmail.com"
    password = "pcgz vspu qhot proo"
    # Create the email content
    subject = f"Stock Alert: {product_name} Below Threshold"
    body = f"The stock of {product_name} is below the threshold. Current stock: {stock_quantity}, Threshold: {threshold}."
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    # Send the email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"Alert email sent for {product_name}")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
"""
# Function to update stock after a successful order
def update_stock(product_name, quantity_ordered):
    global last_email_sent
    stock_data = load_stock_data()
    product_name = product_name.strip().lower()  # Normalize product name
    product_found = False  # Flag to check if the product exists

    for item in stock_data:
        if item['name'].strip().lower() == product_name:  # Normalize product name for comparison
            if item['stock'] >= quantity_ordered:
                item['stock'] -= quantity_ordered
                # Update availability based on remaining stock
                item['availability'] = "In Stock" if item['stock'] > 0 else "Out of Stock"
                product_found = True

                order_details = {
                    "Medicine Name": item['name'],
                    "Ordered Quantity": quantity_ordered,
                    "Pack Size Label": item['pack_size_label'],
                    "Price": item['price'],
                    "Order Date": datetime.now().strftime("%Y-%m-%d"),  # Add current date
                    "Manufacturer Name": item['manufacturer_name']
                }

                # Generate the bill after successful stock update
                generate_order_bill(order_details)

                # If the product becomes out of stock, send an email immediately
                if item['availability'] == "Out of Stock":
                    current_date = datetime.now().date()
                    if product_name not in last_email_sent or last_email_sent[product_name] != current_date:
                        send_out_of_stock_products_email([item['name']], order_details)
                        last_email_sent[product_name] = current_date  # Update the last email sent date
            else:
                return f"Insufficient stock for {item['name']}. Only {item['stock']} items available."
            break

    if not product_found:
        return "Product not found."

    # Write updated stock data back to the CSV file
    try:
        collection.update_one(
            {'name': item['name']},
            {'$set': {
                'stock': item['stock'],
                'availability': item['availability']
            }}
        )
    except Exception as e:
        return f"Error updating stock in MongoDB: {e}"


    return f"Stock updated. New stock for {product_name}: {item['stock']}."

def increase_stock(product_name, quantity):
    product_name = product_name.strip().lower()
    item = collection.find_one({'name': {'$regex': f'^{product_name}$', '$options': 'i'}})

    if not item:
        return f"Sorry, {product_name} not found in inventory."

    current_stock = int(item.get('stock', 0))
    new_stock = current_stock + quantity
    if new_stock <= 0:
        availability = "Out of Stock"
    else:
        availability = "In Stock"

    collection.update_one({'_id': item['_id']}, {'$set': {'stock': new_stock, 'availability': availability}})
    return f"{product_name.capitalize()} stock increased by {quantity}. New stock: {new_stock}"


# Route for Welcome Page
@app.route('/')
def welcome():
    return render_template('welcome.html')

# Route for Home (Logged-in User Info)
@app.route('/home')
def home():
    if 'user_id' in session:
        user = user_collection.find_one({'_id': ObjectId(session['user_id'])})
        if user:
            return render_template('home.html', user=user)
    return redirect(url_for('login'))

user_collection = mongo.db.user_details
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = user_collection.find_one({"username": username})
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])  # Store user's Mongo ID in session
            return redirect(url_for('home'))
        else:
            return render_template('login.html', message="Invalid credentials, try again.")

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Input validation
        if not re.match(r'^[a-zA-Z0-9]+$', username):
            return render_template('register.html', message="Username must contain only letters and numbers.")
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return render_template('register.html', message="Invalid email format.")
        if password != confirm_password:
            return render_template('register.html', message="Passwords do not match.")
        if len(password) < 6:
            return render_template('register.html', message="Password must be at least 6 characters long.")

        # Check if user already exists
        if user_collection.find_one({"$or": [{"username": username}, {"email": email}]}):
            return render_template('register.html', message="Username or Email already exists.")

        hashed_password = generate_password_hash(password)
        user_collection.insert_one({
            "username": username,
            "email": email,
            "password": hashed_password
        })

        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/notification')
def notification():
    stock_data = load_stock_data()
    today = datetime.now().date()

    # Filter: Expiring within 30 days
    expiring_soon = [item for item in stock_data if (item['expiry_date'] - today).days <= 30 and (item['expiry_date'] - today).days >= 0]

    # Filter: Out of stock
    out_of_stock = [item for item in stock_data if item['availability'].lower() == "out of stock"]

    return render_template('notification.html', expiring_soon=expiring_soon, out_of_stock=out_of_stock)

@app.route('/stock-prediction', methods=['GET'])
def stock_prediction():
    return render_template('stock_prediction.html')

# Route for Chatbot Page
@app.route('/chatbot')
def chatbot():
    if 'user_id' in session:
        return render_template('chatbot.html')
    return redirect(url_for('login'))

# Route for Logout
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

# API endpoint for chatbot interaction
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    username = session.get('username')  # Get the username from the session

    # Greeting responses
    greetings = ["hello", "hi", "hey"]
    if user_input.lower() in greetings:
        return jsonify({'response': "Hello! Welcome to Medical Stock Management. How can I assist you today?"})

    # End greeting responses
    end_greetings = ["bye", "goodbye", "exit"]
    if user_input.lower() in end_greetings:
        return jsonify({'response': "Thank you for using Medical Stock Management. Have a great day!"})

    if 'order' in user_input.lower():  # Check if the user is ordering a product
        try:
            words = user_input.lower().split()
            # Extract quantity and product name from the input
            if 'order' in words:
                quantity_index = words.index('order') + 1
                quantity_ordered = int(words[quantity_index])
                product_name = ' '.join(words[quantity_index + 1:]).strip()

                # Check stock availability
                stock_response = get_stock_availability(product_name)
                if 'Sorry' in stock_response:  # Product not found
                    response = stock_response
                elif 'Alert' in stock_response:  # Stock is below threshold
                    response = stock_response
                else:
                    # Update stock after successful order
                    update_response = update_stock(product_name, quantity_ordered)
                    response = f"{stock_response}\n{update_response}"
            else:
                response = "Invalid order format. Please specify the quantity and product name."
        except (ValueError, IndexError) as e:
            response = f"Error processing your order: {str(e)}"
    elif 'purchase' in user_input.lower():
        try:
            words = user_input.lower().split()
            if 'purchase' in words:
                quantity_index = words.index('purchase') + 1
                quantity_purchased = int(words[quantity_index])
                product_name = ' '.join(words[quantity_index + 1:]).strip()

                # Call a utility to increase stock
                update_response = increase_stock(product_name, quantity_purchased)
                response = f"Stock update: {update_response}"
            else:
                response = "Invalid purchase format. Please specify the quantity and medicine name."
        except (ValueError, IndexError) as e:
            response = f"Error processing your purchase: {str(e)}"
    else:
        response = get_stock_availability(user_input)  # Handle normal stock queries

    return jsonify({'response': response})

# Route to trigger daily expiry report manually
@app.route('/send_daily_expiry_report', methods=['GET'])
def send_daily_expiry_report_route():
    try:
        send_daily_expiry_report()
        return jsonify({'message': 'Daily expiry report sent successfully.'})
    except Exception as e:
        return jsonify({'message': f'Error sending daily expiry report: {str(e)}'}), 500

# Route to check stock details
@app.route('/check_stocks')
def check_stocks():
    stock_data = load_stock_data()
    stock_summary = [
        {
            "id": item["id"],
            "name": item["name"],
            "stock": item["stock"],
            "availability": item["availability"],
            "manufacture_date": item["manufacture_date"].strftime('%d-%m-%Y'),  # Include manufacture date
            "expiry_date": item["expiry_date"].strftime('%d-%m-%Y')  # Include expiry date
        }
        for item in stock_data
    ]
    return jsonify({'stock_summary': stock_summary})


@app.route('/predict_stock', methods=['POST'])
def predict_stock():
    # Get input from the frontend 
    data = request.json
    medicine_name = data['medicine_name']
    target_month = pd.to_datetime(data['month'])

    # Get the two previous months' periods
    prev_month1 = (target_month - pd.DateOffset(months=1)).to_period('M')
    prev_month2 = (target_month - pd.DateOffset(months=2)).to_period('M')

    client = MongoClient("mongodb://localhost:27017/")
    db = client['medical_inventory']
    collection = db['order_details1']
    # Load and preprocess data from MongoDB
    df = pd.DataFrame(list(collection.find({"medicine_name": medicine_name})))
    if df.empty:
        return jsonify({'error': 'No data found for this medicine in order'}), 404

    df['order_date'] = pd.to_datetime(df['order_date'])
    df['month'] = df['order_date'].dt.to_period('M')

    # Group by medicine, manufacturer, and pack size to get total order quantity for each month
    grouped = df.groupby(['medicine_name', 'manufacturer_name', 'pack_size_label', 'month'])['order_quantity'].sum().reset_index()

    # Get the data for the previous 2 months
    past1 = grouped[grouped['month'] == prev_month1]
    past2 = grouped[grouped['month'] == prev_month2]

    if past1.empty or past2.empty:
        return jsonify({'error': 'Insufficient past data for prediction'}), 400

    # Get the latest known attributes like price, manufacturer, and pack size
    latest_info = df.drop_duplicates(subset=['medicine_name', 'manufacturer_name', 'pack_size_label']) \
                    [['medicine_name', 'manufacturer_name', 'pack_size_label', 'price']].iloc[0]

    # Encode categorical data for RF model
    med_rf = le_medicine_rf.transform([latest_info['medicine_name']])[0]
    manu_rf = le_manufacturer_rf.transform([latest_info['manufacturer_name']])[0]
    pack_rf = le_pack_size_rf.transform([latest_info['pack_size_label']])[0]

    # Encode categorical data for MLP model
    med_mlp = le_medicine_mlp.transform([latest_info['medicine_name']])[0]
    manu_mlp = le_manufacturer_mlp.transform([latest_info['manufacturer_name']])[0]
    pack_mlp = le_pack_size_mlp.transform([latest_info['pack_size_label']])[0]

    # Get the quantities for past months
    past_month1_qty = past1['order_quantity'].values[0]
    past_month2_qty = past2['order_quantity'].values[0]

    X_rf = pd.DataFrame([[med_rf, manu_rf, pack_rf, float(latest_info['price']), past_month1_qty, past_month2_qty]],
                    columns=['medicine_name_encoded', 'manufacturer_name_encoded',
                             'pack_size_label_encoded', 'price', 'past_month1', 'past_month2'])
    X_mlp = pd.DataFrame([[med_mlp, manu_mlp, pack_mlp, past_month1_qty, past_month2_qty]],
                     columns=['medicine_name_encoded', 'manufacturer_name_encoded',
                              'pack_size_label_encoded', 'past_month1', 'past_month2'])

    
    # Predict using the Random Forest model
    rf_pred = float(rf_model.predict(X_rf)[0])

    # Predict using the MLP model
    X_scaled = scaler.transform(X_mlp)
    mlp_pred_log = mlp_model.predict(X_scaled)
    mlp_pred=float(np.expm1(mlp_pred_log))
  

    # Combine predictions (you can adjust weights)
    final_pred = round((0.7 * rf_pred + 0.3 * mlp_pred), 2)
    print(f"Predicted quantity for {medicine_name}: {final_pred} units")
    print(f"Random Forest Prediction: {rf_pred}")
    print(f"MLP Prediction: {mlp_pred}")

    # Return predictions as JSON
    return jsonify({
        'final_prediction': final_pred
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

