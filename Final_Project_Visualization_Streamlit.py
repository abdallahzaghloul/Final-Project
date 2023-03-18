from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd   
from imblearn.over_sampling import SMOTE
import itertools

from math import ceil #4

import plotly.graph_objects as go  #6
import plotly.express as px  #7
from plotly.subplots import make_subplots  #8
from sklearn.model_selection import train_test_split  #9 


import plotly.figure_factory as ff #21
from category_encoders import BinaryEncoder #23


from sklearn.impute import SimpleImputer  #11
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder , MinMaxScaler , StandardScaler, RobustScaler #12
from sklearn.linear_model import LinearRegression  #13
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier



#Reading & Ecxploting  Data

df = pd.read_csv('Adult.csv', na_values=['N/A', 'no', ' ?','NaN','No info'])
df.columns  = [i.replace(' ','_') for i in df.columns]
df.columns  = [i.upper() for i in df.columns]
df.head(3)




df['TARGET'] = df.TARGET.str.replace('<=50K','0').str.replace('>50K','1')
df['TARGET']=df.TARGET.astype('float')


df.drop(['EDUCATION'], axis=1, inplace=True)
df.dropna(axis=0, inplace= True)



Catego=['WORK_CLASS','MARITAL_STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','NATIVE_COUNTRY']
Numero=['AGE','FNLWGT','EDUCATION_NUMBER','CAPITAL_GAIN', 'CAPITAL_LOSS', 'HRS/WEEK']

fig1=px.histogram(data_frame=df, x=WORK_CLASS)
fig2=px.histogram(data_frame=df, x=MARITAL_STATUS)
fig3=px.histogram(data_frame=df, x=OCCUPATION)
fig4=px.histogram(data_frame=df, x=RELATIONSHIP)
fig5=px.histogram(data_frame=df, x=RACE)
fig6=px.histogram(data_frame=df, x=SEX)
fig7=px.histogram(data_frame=df, x=NATIVE_COUNTRY)
fig8=px.histogram(data_frame=df, x=AGE)
fig9=px.histogram(data_frame=df, x=EDUCATION_NUMBER)
fig10=px.histogram(data_frame=df, x=CAPITAL_GAIN)
fig11=px.histogram(data_frame=df, x=CAPITAL_LOSS)
fig12=px.histogram(data_frame=df, x=HRS/WEEK)
fig13=px.histogram(data_frame=df, x=FNLWGT)

for i in [' Cuba', ' Jamaica', ' India', ' Mexico', ' Puerto-Rico',' Honduras' ,' England' ,' Canada', ' Germany', ' Iran' ,' Philippines', ' Poland', ' Columbia', ' Cambodia', ' Thailand', ' Ecuador', ' Laos', ' Taiwan', ' Haiti', ' Portugal', ' Dominican-Republic' ,' El-Salvador',' France' ,' Guatemala', ' Italy', ' China', ' South' ,' Japan', ' Yugoslavia',' Peru', ' Outlying-US(Guam-USVI-etc)' ,'Guam-USVI',' Scotland', ' Trinadad&Tobago',' Greece', ' Nicaragua', ' Vietnam', ' Hong', ' Ireland' ,' Hungary',' Holand-Netherlands']:
    df.NATIVE_COUNTRY=df.NATIVE_COUNTRY.str.replace(i,'Other')


df.NATIVE_COUNTRY=df.NATIVE_COUNTRY.str.replace(' United-States','USA')  
df.NATIVE_COUNTRY=df.NATIVE_COUNTRY.str.replace(' ','')



fig5 = px.imshow(df.corr()) 





X = df.drop('TARGET', axis=1)
y = df.TARGET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Catego_traino =  X_train.select_dtypes(include=['object']).columns
Catego_traino



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

# Logistic Regression
# Logistic Regression After SMOTE
lr = LogisticRegression(random_state=42)
lr.fit(X_train_sm, y_train_sm)
y_pred = lr.predict(X_test_enc)

#fig_cm1 = plot_confusion_matrix(lr, X_test_enc, y_test, cmap='Blues', values_format='d')



rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_sm, y_train_sm)
y_pred = rf.predict(X_test_enc)

#fig_cm2= plot_confusion_matrix(rf, X_test_enc, y_test, cmap='Blues', values_format='d')


# Feature Importance
importances = pd.DataFrame({'feature':X_train_enc.columns, 'importance':np.round(rf.feature_importances_,3)})
importances = importances.sort_values('importance', ascending=False).set_index('feature')




# Grid Search
# Grid Search
# Grid Search
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100,140,130,120], 'max_depth': [60,50,70]}

grid = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train_sm, y_train_sm)

print(grid.best_params_)




rf = RandomForestClassifier(random_state=42, n_estimators=grid.best_params_['n_estimators'], max_depth=grid.best_params_['max_depth'])
rf.fit(X_train_sm, y_train_sm)
y_pred = rf.predict(X_test_enc)

#fig_cm3=plot_confusion_matrix(rf, X_test_enc, y_test, cmap='Blues', values_format='d')


import pickle

# Save the model as a pickle in a file

filename = 'Adult_Model.sav'

pickle.dump(rf, open(filename, 'wb')) 


#def predict(user):

    # Load the model from the file
#    model = pickle.load(open(filename, 'rb'))

    # Make predictions
#    predictions = model.predict(user)

#    return predictions
    
st.markdown(" <center>  <h1> Predicting Adult's Annual Salary </h1> </font> </center> </h1> ",
            unsafe_allow_html=True)

im = Image.open("50K$.jpg")
image = np.array(im)
st.image(image)








