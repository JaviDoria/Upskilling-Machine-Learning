import streamlit as st
import pickle
import pandas as pd

# Cargar el modelo y el escalador
    with open('LogisticRegression.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# Título de la aplicación
st.title("Predicción de Suscripción a Depósitos a Plazo")

# Entrada de datos del usuario
st.header("Introduce los datos del cliente:")

age = st.number_input("Edad:", min_value=16, max_value=125, value=30)
education = st.radio("Nivel educativo:", ['primaria', 'secundaria', 'terciaria'])
default = st.radio("Tiene crédito en mora:", ['no', 'sí'])
balance = st.number_input("Saldo de la cuenta (en euros):", value=0)
housing = st.radio("¿Tiene hipoteca?", ['no', 'sí'])
loan = st.radio("¿Tiene un préstamo personal?", ['no', 'sí'])
pdays = st.number_input("Días desde el último contacto con la campaña (pdays):", value=-1)

job = st.selectbox("Tipo de trabajo:", 
                   ['office', 'other', 'self-employed', 'service', 'student', 
                    'unemployed', 'nan'])
marital = st.radio("Estado civil:", ['married', 'single'])

# Crear DataFrame con los datos del usuario
user_data = pd.DataFrame({
    'age': [age],
    'education': [education],
    'default': [default],
    'balance': [balance],
    'housing': [housing],
    'loan': [loan],
    'pdays': [pdays],
    'job': [job],
    'marital': [marital]
})

# Procesar los datos
user_data['default'] = user_data['default'].map({'no': 0, 'sí': 1}).astype(int)
user_data['housing'] = user_data['housing'].map({'no': 0, 'sí': 1}).astype(int)
user_data['loan'] = user_data['loan'].map({'no': 0, 'sí': 1}).astype(int)
user_data['education'] = user_data['education'].map({'primaria': 1, 'secundaria': 2, 'terciaria': 3}).astype(int)

# Codificar variables categóricas
user_encoded_data = pd.get_dummies(user_data, columns=['job', 'marital'])

# Asegurar que las columnas necesarias están presentes
required_columns = [
    'age', 'education', 'default', 'balance', 'housing', 'loan', 'pdays',
    'job_office', 'job_other', 'job_self-employed', 'job_service', 'job_student',
    'job_unemployed', 'job_nan', 'marital_married', 'marital_single'
]
for col in required_columns:
    if col not in user_encoded_data.columns:
        user_encoded_data[col] = 0
user_encoded_data = user_encoded_data[required_columns]

# Estandarizar las columnas necesarias
scale_variables = ['age', 'balance', 'pdays']
user_encoded_data[scale_variables] = scaler.transform(user_encoded_data[scale_variables])

# Realizar la predicción
prediction = model.predict(user_encoded_data)

# Mostrar el resultado
st.header("Resultado de la Predicción:")
if prediction[0] == 1:
    st.success("El cliente probablemente SE SUSCRIBIRÁ a un depósito a plazo.")
else:
    st.error("El cliente probablemente NO SE SUSCRIBIRÁ a un depósito a plazo.")
