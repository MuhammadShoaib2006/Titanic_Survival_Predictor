import streamlit as st
import pandas as pd
import requests
import math

# Page configuration
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Prediction")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app predicts Titanic survival using machine learning.
    
    **Model Features:**
    - Passenger class
    - Age & gender
    - Family members
    - Ticket fare
    - Embarkation port
    
    Model accuracy: ~80%
    """)

# Input form
with st.form("prediction_form"):
    st.header("Passenger Details")
    
    # First row
    col1, col2, col3 = st.columns(3)
    with col1:
        pclass = st.selectbox("Class", ("First", "Second", "Third"), index=0)
    with col2:
        sex = st.selectbox("Gender", ("Male", "Female"), index=0)
    with col3:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
    
    # Second row
    st.header("Family Information")
    col4, col5 = st.columns(2)
    with col4:
        sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=10, value=0)
    with col5:
        parch = st.number_input("Parents/Children", min_value=0, max_value=10, value=0)
    
    # Third row
    st.header("Journey Details")
    col6, col7 = st.columns(2)
    with col6:
        fare = st.number_input("Fare", min_value=0.0, value=32.0, step=1.0)
    with col7:
        embarked = st.selectbox(
            "Embarked From", 
            ("Cherbourg", "Queenstown", "Southampton"), 
            index=2
        )
    
    submitted = st.form_submit_button("Predict Survival")

if submitted:
    # Prepare data for API
    pclass_map = {"First": 1, "Second": 2, "Third": 3}
    sex_map = {"Male": 0, "Female": 1}
    embarked_map = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
    
    input_data = {
        "Pclass": pclass_map[pclass],
        "Sex": sex_map[sex],
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked_map[embarked]
    }
    
    # Call Flask API
    try:
        response = requests.post("http://localhost:5000/predict", json=input_data)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            
            if prediction == 1:
                st.success("🎉 This passenger would likely survive!")
                st.balloons()
            else:
                st.error("💔 This passenger would likely not survive")
                st.snow()
        else:
            st.error(f"API Error: {response.text}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to API: {str(e)}")