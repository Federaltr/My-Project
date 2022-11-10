import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder


st.sidebar.title("Car Price Prediction")
html_temp = """
<div style="background-color:orange;padding:10px">
<h2 style="color:white;text-align:center;">Federal Price Prediction App with Streamlit </h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)


age=st.sidebar.selectbox("What is the age of your car:",(0,1,2,3))
hp=st.sidebar.slider("What is the hp_kw of your car?", 40, 300, step=5)
km=st.sidebar.slider("What is the km of your car", 0,350000, step=1000)
gearing_type=st.sidebar.radio('Select gear type',('Automatic','Manual','Semi-automatic'))
car_model=st.sidebar.selectbox("Select model of your car", ('Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'))


heagle_model = pickle.load(open(r"C:\Users\federal\Desktop\GitHub\My-Project\DataScience\ML_Deployment\CarPricePredictionApp\model_new","rb"))
heagle_transformer = pickle.load(open('transformer', 'rb'))


my_dict = {
    "age": age,
    "hp_kW": hp,
    "km": km,
    'gearing_type':gearing_type,
    "make_model": car_model
    
}

df = pd.DataFrame.from_dict([my_dict])


st.header("Selected information is below")
st.table(df)

df2 = heagle_transformer.transform(df)

st.subheader("Press predict")

if st.button("Predict"):
    prediction = heagle_model.predict(df2)
    st.success("Price of your car is €{}. ".format(int(prediction[0])))
    
