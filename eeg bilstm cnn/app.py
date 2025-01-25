import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
#from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the models
cnn_model = tf.keras.models.load_model('cnn_model.h5')
lstm_model = tf.keras.models.load_model('lstm_model.h5')

# Function to make predictions
def predict_seizure(model, data):
    data = np.expand_dims(data, axis=2)  # Reshape for model input
    if data.shape[1] != 178:  # Check if the number of features is as expected
        raise ValueError(f"Expected 178 features but got {data.shape[1]}.")
    prediction = model.predict(data)
    return np.argmax(prediction, axis=1)

# Streamlit app
st.title('Seizure Prediction App')

st.write("""
### Upload EEG Data
Please upload a CSV or TXT file containing the EEG data for prediction.
""")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])

# Initialize data variable
data = None

if uploaded_file is not None:
    try:
        # Determine file type and read data accordingly
        file_type = uploaded_file.name.split('.')[-1]
        
        if file_type == "csv":
            data = pd.read_csv(uploaded_file, header=None)
        elif file_type == "txt":
            # Assuming space or tab-separated values in TXT file
            data = pd.read_csv(uploaded_file, sep='\s+', header=None)  # Adjust separator if needed

        if data is not None:
            # Handle missing values
            data = data.fillna(0)  # Replace NaNs with 0
            
            # Display data to check
            st.write("Data Preview:")
            st.write(data.head())
            
            # Convert to numeric and ensure correct shape
            data = pd.to_numeric(data.stack(), errors='coerce').unstack()  # Convert all data to numeric, coercing errors to NaN
            data = data.fillna(0).astype(int)  # Replace NaNs with 0 and convert to integers
            
            if data.shape[1] != 178:
                st.error(f"Data should have 178 features but has {data.shape[1]}.")
                st.stop()

            model_choice = st.selectbox("Select Model for Prediction", ["CNN", "BiLSTM"])
            
            if model_choice == "CNN":
                model = cnn_model
            else:
                model = lstm_model
            
            if st.button('Predict'):
                try:
                    predictions = predict_seizure(model, data)
                    
                    st.write("Prediction Results:")
                    st.write(pd.DataFrame(predictions, columns=['Prediction']).replace({0: 'Healthy', 1: 'Epileptic'}))
                    
                    st.write("Prediction Distribution:")
                    st.bar_chart(pd.Series(predictions).replace({0: 'Healthy', 1: 'Epileptic'}).value_counts())

                    st.write("Prediction Complete. Below is the distribution of predicted classes:")

                    # Show Prediction Distribution
                    fig, ax = plt.subplots()
                    pd.Series(predictions).replace({0: 'Healthy', 1: 'Epileptic'}).value_counts().plot(kind='bar', ax=ax)
                    st.pyplot(fig)
                except ValueError as e:
                    st.error(f"Error during prediction: {e}")
        else:
            st.error("No data loaded from file.")
    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
else:
    st.write("Please upload a file.")