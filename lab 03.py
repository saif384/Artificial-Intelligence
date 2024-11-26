import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv('Electronics_data.csv')
print("No of data rows:", data.shape[0])

# Normalization for numerical columns
scalar = MinMaxScaler()
data[['Customer_ID', 'Age', 'Total_Price']] = scalar.fit_transform(data[['Customer_ID', 'Age', 'Total_Price']])

# Standardization for 'Quantity'
standard_scalar = StandardScaler()
data[['Quantity']] = standard_scalar.fit_transform(data[['Quantity']])

# One-hot encoding for categorical variables
one_hot_encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
one_hot_encoded = one_hot_encoder.fit_transform(data[['Gender', 'Loyalty_Member']])
data = pd.concat([data.drop(columns=['Gender', 'Loyalty_Member']), one_hot_encoded], axis=1)


order_status = data['Order_Status'].copy()  
data = data.drop(columns=['Order_Status'])

# Ordinal encoding for Product_Type, Payment_Method, Shipping_Type, Order_Status
product_type_categories = [['Smartphone', 'Tablet', 'Laptop', 'Headphones', 'Smartwatch']]
ordinal_encoder_product = OrdinalEncoder(categories=product_type_categories)
data['Product_Type'] = ordinal_encoder_product.fit_transform(data[['Product_Type']])

payment_method_categories = [['Credit Card','Paypal', 'Cash', 'Debit Card', 'Bank Transfer']]
ordinal_encoder_payment = OrdinalEncoder(categories=payment_method_categories, handle_unknown='use_encoded_value', unknown_value=-1)
data['Payment_Method'] = ordinal_encoder_payment.fit_transform(data[['Payment_Method']])

shipping_type_category = [['Standard', 'Overnight', 'Express','Expedited', 'Same Day']]
ordinal_encoder_shipping = OrdinalEncoder(categories=shipping_type_category)
data['Shipping_Type'] = ordinal_encoder_shipping.fit_transform(data[['Shipping_Type']]) 

order_status_categories = [['Cancelled', 'Completed']]
ordinal_encoder_status = OrdinalEncoder(categories=order_status_categories)
encoded_order_status = ordinal_encoder_status.fit_transform(order_status.values.reshape(-1, 1))
data['Order_Status'] = encoded_order_status

# Binning for Unit_Price
data['Unit_Price'] = pd.qcut(data['Unit_Price'], q=3, labels=False)

data.to_csv('processed_data.csv', index=False)

x = data  
y = order_status  

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("No. of training rows:", x_train.shape[0])
print("No. of testing rows:", x_test.shape[0])

# Initialize and train the MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp_classifier.fit(x_train, y_train)
# Make predictions
y_pred = mlp_classifier.predict(x_test)
# Print model details and evaluation metrics
print("Hidden_layer_size:", mlp_classifier.hidden_layer_sizes)
print("No. of layers:", mlp_classifier.n_layers_)
print("No. of iterations:", mlp_classifier.n_iter_)
print("Classes:", mlp_classifier.classes_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))