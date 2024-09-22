# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load the dataset
data = pd.read_csv('diabetes_prediction_dataset.csv')


# %%
# Check for missing values
data = data.dropna()

# Encode categorical features using LabelEncoder
le = LabelEncoder()

# List of categorical columns to encode
categorical_columns = ['gender', 'smoking_history']

for col in categorical_columns:
    data[col] = le.fit_transform(data[col])


# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split features and target
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')

# %%
# Build the TensorFlow model
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
model.save('diabetes_model.h5')  # Save the trained model
# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)


def predict_new_data(input_data):
    input_data = scaler.transform(input_data)  # Scale input
    prediction = model.predict(input_data)
    return prediction[0][0]





def predict_new_data(input_data):
    # Ensure input_data is a DataFrame and has the same structure as the original training data
    input_data = pd.DataFrame(input_data)  # Convert to DataFrame

    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Return the prediction (as a float)
    return prediction[0][0]