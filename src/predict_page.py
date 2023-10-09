import streamlit as st
import pickle
import numpy as np

def load_model():
    with open("./src/saved_steps.pkl", "rb") as file:
        data = pickle.load(file)
    return data
data = load_model()

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")
    country = st.selectbox("Country",
                 ('United Kingdom of Great Britain and Northern Ireland',
       'Netherlands', 'United States of America', 'Italy', 'Canada',
       'Germany', 'Poland', 'France', 'Brazil', 'Sweden', 'Spain',
       'India', 'Australia'), index=None, placeholder="Select Country of Employment" )
    education = st.selectbox("Enducation Level", ('Master’s degree', "Bachelor's degree", 'Less than a Bachelors',
       'Post grad'), index=None, placeholder="Select Education Level")
    
    experience = st.slider("Years of Experience", 1, 9, 3)
   

    ok = st.button("Predict")
    prediction=0.0
    if ok:
        response = np.array([[country, education, experience]])
        response[:, 0] = le_country.transform(response[:, 0])
        response[:, 1] = le_education.transform(response[:, 1])
        response = response.astype(float)
        prediction = regressor_loaded.predict(response)
        st.subheader("Developer Salary is ${:,.2f}".format(prediction[0]))
    
    