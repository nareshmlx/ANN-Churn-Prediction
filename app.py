import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

## load the encoder and scaler
with open('label_enc_gender.pkl','rb') as file:
    label_enc_gender = pickle.load(file)
    
with open('one_hot_geo.pkl','rb') as file:
    one_hot_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
  scaler = pickle.load(file)
  
st.title("Customer Churn Prediction")

geography = st.selectbox('Geography', one_hot_geo.categories_[0])
gender = st.selectbox('Gender', label_enc_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_enc_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


geo_encoded = one_hot_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns = one_hot_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_scaled = scaler.transform(input_data)
# Assuming input_data_scaled is already defined and the model is loaded

# Make a prediction
prediction = model.predict(input_data_scaled)
pred = prediction[0][0]

# Streamlit button for making predictions
if st.button('Predict'):
    # Check if the customer is likely to churn
    if pred > 0.5:
        st.write('**The customer is likely to churn**')
        st.write(f'Churn Probability: **{pred * 100:.2f}%**')
    else:
        st.write('**The customer is likely to stay**')
        st.write(f'Stay Probability: **{(1 - pred) * 100:.2f}%**')











