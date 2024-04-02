#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Import necessary libraries
import streamlit as st
import pandas as pd
!pip install scikit-learn
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the trained machine learning model
with open('base.pkl', 'rb') as f:  # Replace 'your_model_file.pkl' with the path to your trained model file
    model = pickle.load(f)
data=pd.read_csv(r"C:\Users\Varshini\Downloads\Disease_symptom_and_patient_profile_dataset (1).csv")
# Define the function to preprocess input data
def preprocess_input(df):
    # Perform any preprocessing steps such as encoding categorical variables
    le = LabelEncoder()
    df['Disease']=le.fit_transform(df['Disease'])
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Fever'] = df['Fever'].map({'Yes': 1, 'No': 0})
    df['Cough'] = df['Cough'].map({'Yes': 1, 'No': 0})
    df['Fatigue'] = df['Fatigue'].map({'Yes': 1, 'No': 0})
    df['Difficulty Breathing'] = df['Difficulty Breathing'].map({'Yes': 1, 'No': 0})
    df['Blood Pressure'] = df['Blood Pressure'].map({'Low': 0, 'Normal': 1,'High':2})
    df['Cholesterol Level'] = df['Cholesterol Level'].map({'Low': 0, 'Normal': 1,'High':2})
    # Drop irrelevant columns
   
    return df

# Define the title of the web app
st.title('Disease Prediction App')

# Create a sidebar for user input
st.sidebar.header('User Input')

# Create input fields for each feature
fever = st.sidebar.selectbox('Fever', ['Yes', 'No'])
cough = st.sidebar.selectbox('Cough', ['Yes', 'No'])
fatigue = st.sidebar.selectbox('Fatigue', ['Yes', 'No'])
difficulty_breathing = st.sidebar.selectbox('Difficulty Breathing', ['Yes', 'No'])
age = st.sidebar.slider('Age', 0, 100, 25)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
blood_pressure = st.sidebar.selectbox('Blood Pressure', ['Low', 'Normal', 'High'])
cholesterol_level = st.sidebar.selectbox('Cholesterol Level', ['Low', 'Normal', 'High'])
Disease = st.sidebar.selectbox('Select Disease', data['Disease'].unique())

# Filter dataset based on selected disease
filtered_data = data[data['Disease'] == selected_disease]

# Create a dataframe with the user input
input_data = pd.DataFrame({'Disease':[Disease],
    'Fever': [fever],
                           'Cough': [cough],
                           'Fatigue': [fatigue],
                           'Difficulty Breathing': [difficulty_breathing],
                           'Age': [age],
                           'Gender': [gender],
                           'Blood Pressure': [blood_pressure],
                           'Cholesterol Level': [cholesterol_level]})

# Preprocess the input data
input_data = preprocess_input(input_data)

# Make predictions
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Display prediction
st.subheader('Prediction')
st.write(prediction[0])

# Display prediction probability
st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[2]:


get_ipython().system('pip install streamlit')
get_ipython().system('pip install scikit-learn')
# In[ ]:




