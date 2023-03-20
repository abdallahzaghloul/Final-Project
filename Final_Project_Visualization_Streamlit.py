from PIL import Image
import numpy as np
import pandas as pd   
import streamlit as st
import pickle
#Reading & Ecxploting  Data

    
st.markdown(" <center>  <h1> Predicting Adult's Annual Salary </h1> </font> </center> </h1> ",
            unsafe_allow_html=True)

#im = Image.open("C://Users//hp//Desktop//Data Science//Final Project//50K$.jpg")
im = Image.open("50K$.jpg")
image = np.array(im)
st.image(image)

Age=st.slider('Select your age',1,100)
Education_Number = st.slider('Select your education years',1,25)

HRS_WK = st.slider('How many hours are you supposed to work per year',1,50)
 
Race=st.selectbox("Select your race",(' White',' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo' ,' Other'))
Sex=st.selectbox("Select your race",(' Male' ,' Female'))

Native_Country = st.selectbox("Select your country",('USA' ,'Other'))
 
Occupation = st.selectbox("Select your occupation",(' Adm-clerical' ,' Exec-managerial',' Handlers-cleaners',' Prof-specialty',' Other-service' ,' Sales' ,' Transport-moving' ,' Farming-fishing',' Machine-op-inspct', ' Tech-support' ,' Craft-repair', ' Protective-serv',' Armed-Forces' ,' Priv-house-serv'))
 
Capital_Gain = st.number_input('Enter the capital gain') 
Capital_Loss = st.number_input('Enter the capital loss') 


Relationship = st.selectbox("Select your relationship status",(' Not-in-family' ,' Husband' ,' Wife', ' Own-child' ,' Unmarried',' Other-relative'))
Workclass = st.selectbox("Select your workclass",(' State-gov' ,' Self-emp-not-inc', ' Private' ,' Federal-gov', ' Local-gov',' Self-emp-inc' ,' Without-pay' ))
 
Marital_Status =st.selectbox("Select your mzrital status",(' Never-married' ,' Married-civ-spouse', ' Divorced',' Married-spouse-absent' ,' Separated', ' Married-AF-spouse',' Widowed'))
  

#df1 = pd.read_csv('C:/Users/hp/Desktop/Data Science/Final Project/Adult_Ans.csv', na_values=['N/A', 'no', ' ?','NaN','No info'])
df1 = pd.read_csv('Adult_Ans.csv', na_values=['N/A', 'no', ' ?','NaN','No info'])
df1.columns  = [i.replace(' ','_') for i in df1.columns]
df1.columns  = [i.upper() for i in df1.columns]



df1['AGE'].iloc[0]=Age
df1['EDUCATION_NUMBER'].iloc[0]=Education_Number
df1['HRS/WEEK'].iloc[0]=HRS_WK
df1['RACE'].iloc[0]=Race
df1['SEX'].iloc[0]=Sex
df1['NATIVE_COUNTRY'].iloc[0]=Native_Country
df1['OCCUPATION'].iloc[0]=Occupation
df1['CAPITAL_GAIN'].iloc[0]=Capital_Gain
df1['CAPITAL_LOSS'].iloc[0]=Capital_Loss
df1['RELATIONSHIP'].iloc[0]=Relationship
df1['WORK_CLASS'].iloc[0]=Workclass
df1['MARITAL_STATUS'].iloc[0]=Marital_Status

Catego1=['WORK_CLASS','MARITAL_STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','NATIVE_COUNTRY']
Numero1=['AGE','EDUCATION_NUMBER','CAPITAL_GAIN', 'CAPITAL_LOSS', 'HRS/WEEK']

filename = 'Adult.sav'
encoder_filename = 'Adult_Encoder.sav'
scaler_filename = 'Adul_Scaler.sav'

encoder = pickle.load(open('Adult_Encoder.pkl', 'rb'))

# Loading scaler
scaler = pickle.load(open('Adul_Scaler.pkl', 'rb'))

# loading the model
model = pickle.load(open('Adult.pkl', 'rb'))

# A function to transform the user input
def transform_user(user):
    # Transform the user input
    user_enc = encoder.transform(user)
    user_enc[Numero1] = scaler.transform(user_enc[Numero1])

    return user_enc

def predict(user):

    # Load the model from the file
    model = pickle.load(open(filename, 'rb'))

    # Make predictions
    predictions = model.predict(user)

    return predictions



if st.button('Calculate'):
    user_enc = transform_user(df1)
    predictions = model.predict(user_enc)
    if predictions ==1:
        st.write(f'The expected salary is to be >50K')
    else :
        st.write(f'The expected salary is to be <=50K')















