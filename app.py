import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

pclass=st.selectbox('Enter the Passenger Class: ',options=[1,2,3])
sex=st.selectbox('Enter the Gender of the Passenger: ',options=['Male','Female'])
age=st.number_input('Enter the Age:',min_value=0.0,max_value=114.0,step=1.0)
sibsp=st.slider('No. of Siblings/Spouse onboard:',min_value=0,max_value=8,step=1)
parch=st.slider('No. of Parent/Children onboard:',min_value=0,max_value=6,step=1)
fare=st.number_input('Enter the fare: ',min_value=0.0,max_value=600.0,step=1.0)
embarked=st.selectbox('Enter the Embarked Place:',options=['Chebourg','Queenstown','Southempton'])

user={'Pclass':pclass, 'Sex':sex.lower(), 'Age':age, 'SibSp':sibsp, 'Parch':parch, 'Fare':fare, 'Embarked':embarked}

user=pd.DataFrame([user])

#load
with open('label_encoder.pkl','rb') as file:
    label=pickle.load(file)
with open('onehot_encoder.pkl','rb') as file:
    onehot=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scale=pickle.load(file)

num=['Age','Fare']    
user['Sex']=label.transform(user['Sex'])
user[num]=scale.transform(user[num])
embarked=onehot.transform(user[['Embarked']])
embark=pd.DataFrame(embarked,columns=onehot.get_feature_names_out())
user=pd.concat([user,embark],axis=1)
user.drop(columns=['Embarked'],inplace=True)

model=load_model('model.h5')

if st.button('Predict Survival Chance'):
    
    y=model.predict(user)
    probability=y[0][0]
    st.write(f"The probability of Survival is: {probability:.2%} ")
    if probability>0.55:
        st.success("Passenger will survive.")
    else:
        st.error('Passenger will not survive.')    
      



