import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import save_model
import tensorflow as tf

# Set Streamlit page config
st.set_page_config(page_title="NeuroNexus - Diabetes Predictor", layout="centered")

# Title
st.title("🧠 NeuroNexus - Diabetes Risk Predictor")
st.markdown("Enter patient details below to check diabetes risk.")

# Load and preprocess data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=columns)
    return df

# Train model or load if exists
@st.cache_resource
def load_model_and_scaler():
    df = load_data()
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    if os.path.exists("diabetes_model.h5"):
        model = load_model("diabetes_model.h5")
    else:
        model = Sequential([
            Dense(32, activation='relu', input_shape=(8,)),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        model.save("diabetes_model.h5")

    return model, scaler

# Load model and scaler
model, scaler = load_model_and_scaler()

# Collect user input
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 300, 120)
        bp = st.number_input("Blood Pressure", 0, 200, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)

    with col2:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 1, 120, 30)

    submit = st.form_submit_button("Predict Risk")

# Predict on submit
if submit:
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]

    if prediction >= 0.5:
        st.error(f"⚠ High risk of diabetes ({prediction:.2%} confidence)")
    else:
        st.success(f"✅ Low risk of diabetes ({(1 - prediction):.2%} confidence)")
