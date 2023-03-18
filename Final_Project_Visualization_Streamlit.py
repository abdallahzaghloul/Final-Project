from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd   
from imblearn.over_sampling import SMOTE
import streamlit as st

import plotly.graph_objects as go  #6
import plotly.express as px  #7
from plotly.subplots import make_subplots  #8
from sklearn.model_selection import train_test_split  #9 


from category_encoders import BinaryEncoder #23


from sklearn.impute import SimpleImputer  #11
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder , MinMaxScaler , StandardScaler, RobustScaler #12
from sklearn.linear_model import LinearRegression  #13
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



#Reading & Ecxploting  Data

#df = pd.read_csv('C:/Users/hp/Desktop/Data Science/Final Project/Adult.csv', na_values=['N/A', 'no', ' ?','NaN','No info'])
df = pd.read_csv('Adult.csv', na_values=['N/A', 'no', ' ?','NaN','No info'])
df.columns  = [i.replace(' ','_') for i in df.columns]
df.columns  = [i.upper() for i in df.columns]
df.head(3)




df['TARGET'] = df.TARGET.str.replace('<=50K','0').str.replace('>50K','1')
df['TARGET']=df.TARGET.astype('float')


df.drop(['EDUCATION','FNLWGT'], axis=1, inplace=True)
df.dropna(axis=0, inplace= True)



Catego=['WORK_CLASS','MARITAL_STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','NATIVE_COUNTRY']
Numero=['AGE','EDUCATION_NUMBER','CAPITAL_GAIN', 'CAPITAL_LOSS', 'HRS/WEEK']



for i in [' Cuba', ' Jamaica', ' India', ' Mexico', ' Puerto-Rico',' Honduras' ,' England' ,' Canada', ' Germany', ' Iran' ,' Philippines', ' Poland', ' Columbia', ' Cambodia', ' Thailand', ' Ecuador', ' Laos', ' Taiwan', ' Haiti', ' Portugal', ' Dominican-Republic' ,' El-Salvador',' France' ,' Guatemala', ' Italy', ' China', ' South' ,' Japan', ' Yugoslavia',' Peru', ' Outlying-US(Guam-USVI-etc)' ,'Guam-USVI',' Scotland', ' Trinadad&Tobago',' Greece', ' Nicaragua', ' Vietnam', ' Hong', ' Ireland' ,' Hungary',' Holand-Netherlands']:
    df.NATIVE_COUNTRY=df.NATIVE_COUNTRY.str.replace(i,'Other')


df.NATIVE_COUNTRY=df.NATIVE_COUNTRY.str.replace(' United-States','USA')  
df.NATIVE_COUNTRY=df.NATIVE_COUNTRY.str.replace(' ','')






X = df.drop('TARGET', axis=1)
y = df.TARGET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Catego_traino =  X_train.select_dtypes(include=['object']).columns



encoder = BinaryEncoder(cols=Catego)
X_train_enc = encoder.fit_transform(X_train)     
X_test_enc = encoder.transform(X_test) 


# Scaling                                                                              
from sklearn.preprocessing import MinMaxScaler                                         
                                                                                       
scaler = MinMaxScaler()                                                                
                                                                                       
X_train_enc[Numero] = scaler.fit_transform(X_train_enc[Numero])
X_test_enc[Numero] = scaler.transform(X_test_enc[Numero])      
                                                                                       

# Hanling imbalanced data
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=32)

X_train_sm, y_train_sm = smote.fit_resample(X_train_enc, y_train)



# Random Forest Before SMOTE
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_sm, y_train_sm)



# Grid Search
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100,140,130,120], 'max_depth': [60,50,70]}

grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train_sm, y_train_sm)


rf = RandomForestClassifier(random_state=42, n_estimators=grid.best_params_['n_estimators'], max_depth=grid.best_params_['max_depth'])
rf.fit(X_train_sm, y_train_sm)
    
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

st.write(df1)
Catego1=['WORK_CLASS','MARITAL_STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','NATIVE_COUNTRY']
Numero1=['AGE','EDUCATION_NUMBER','CAPITAL_GAIN', 'CAPITAL_LOSS', 'HRS/WEEK']

df1_=encoder.transform(df1) 
df1_=scaler.transform(df1_[Numero1])

Y = rf.predict(df1_)


if st.button('Calculate'):
    if Y==1:
        st.write(f'The expected salary would be >50K ')
    
    else: 
        st.write(f'The expected salary would be less than or = 50 K ')














