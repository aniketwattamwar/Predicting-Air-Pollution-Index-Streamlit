 



import numpy as np
import pandas as pd 
import matplotlib as plt
import time
from datetime import datetime
import streamlit as st
 

st.title('Predicting Air Pollution Index')
st.header('Analytics Vidhya Practice Problem')

@st.cache(allow_output_mutation=True)
def load_train_data():
    train_data = pd.read_csv("train.csv")
    return train_data

@st.cache(allow_output_mutation=True)
def load_test_data():
    test_data = pd.read_csv('test.csv')
    return test_data

train_data = load_train_data()
test_data  =load_test_data()
y = train_data.iloc[:,12].values
st.text('Training Data')
st.write(train_data)

st.text('Testing Data')
st.write(test_data)

st.text('Air Pollution Index Column')
st.write(y)
train_data = train_data.drop(['air_pollution_index'],axis = 1)

#Handling the datetime column for training dataset
#Splitting the date time and considering only the time
train_data[['date','time']] = train_data.date_time.str.split(expand=True)
train_data = train_data.drop(['date_time'],axis = 1)
train_data['time'] = train_data['time'].str.replace('[^\w\s]','')
train_data['time']=train_data['time'].astype(float)
train_data = train_data.drop(['date'],axis = 1)
 

train_data = pd.get_dummies(train_data, columns=["weather_type"]) 
train_data = train_data.drop(['weather_type_Squall'],axis = 1)
train_data = pd.get_dummies(train_data, columns=["is_holiday"]) 

train = train_data.iloc[:,:].values
st.subheader('Encoded Training Data')
st.write(train)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_normalized = sc_X.fit_transform(train)

if st.button('Normalized form of Training Data'):
    st.write(train_normalized)

#Handling the datetime column for test dataset
#Splitting the date time and considering only the time
test_data[['date','time']] = test_data.date_time.str.split(expand=True)
test_data = test_data.drop(['date_time'],axis = 1)
test_data['time'] = test_data['time'].str.replace('[^\w\s]','')
test_data['time']=test_data['time'].astype(float)
test_data = test_data.drop(['date'],axis = 1)
 
#Categorical to Numerical
test_data = pd.get_dummies(test_data, columns=["weather_type"]) 
test_data = pd.get_dummies(test_data, columns=["is_holiday"]) 
 
test = test_data.iloc[:,:].values
test_normalized = sc_X.transform(test)


import seaborn as sns
train_data1 = pd.read_csv("train.csv")
corr = train_data1.corr()
st.subheader('Correlation between columns')
st.write(corr)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

 #target column i.e price range
 


#                                       RF using Normalisation | accuracy = 91.45863
#                       n_estimators = 50, log2, acc = 91.46870
#                        n_estimators = 50, log2, acc =     91.48331 - considering time

from sklearn.ensemble import RandomForestRegressor
x = st.slider('Choose number of estimators for Random Forest Algorithm',min_value = 10, max_value = 150)
regressorRF_Norm = RandomForestRegressor(n_estimators = x, random_state = 0,max_features = "log2",oob_score = True)
if st.button('Train Random Forest model'):
    
    regressorRF_Norm.fit(train_normalized, y)

    y_pred_rf_Norm = regressorRF_Norm.predict(test_normalized)
    st.subheader('Predictions are:')
    st.write(y_pred_rf_Norm)

#
#out_norm = pd.DataFrame(y_pred_rf_Norm,columns=['air_pollution_index'])
#out_norm.to_csv('submission_norm_t.csv',sep=',')

#                                              ends here      



                                                
#                                            RF regressor without Normalisation | accuracy = 91.38
from sklearn.ensemble import RandomForestRegressor
regressorRF = RandomForestRegressor(n_estimators = 20, random_state = 0,max_features = "log2",oob_score = True)
regressorRF.fit(train, y)

ypred_rf = regressorRF.predict(test)

out = pd.DataFrame(ypred_rf, columns=['air_pollution_index'])
out.to_csv('submission_rf.csv',sep=',')

#                                            RF regressor ends here 

 
#                                       using PCA | accuracy = 91.23
from sklearn.ensemble import RandomForestRegressor
if st.button('Perform PCA and Train the Random Forest model'):
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 3)
    train_pca = pca.fit_transform(train)
    variance_pca = pca.explained_variance_ratio_ 
    st.text('Variance:')
    st.write(variance_pca)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 3)
    test_pca = pca.fit_transform(test)
    variance_pca = pca.explained_variance_ratio_
    st.text('Variance:')
    st.write(variance_pca)


    
#    value = st.slider('Choose number of estimators ',min_value = 10, max_value = 150)
    regressorRF = RandomForestRegressor(n_estimators = 30, random_state = 0,max_features = "log2",oob_score = True)
#    if st.button('Train model'):
        
    regressorRF.fit(train_pca, y)
     
    ypred_pca = regressorRF.predict(test_pca)
    st.subheader('Predictions are:')
    st.write(ypred_pca)

#out_pca = pd.DataFrame(ypred_pca, columns=['air_pollution_index'])
#out_pca.to_csv('submission_pca.csv',sep=',')

#                               PCA ends here