# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:01:41 2022

@author: ACER
"""

import pickle,os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score

MODEL_SAVE_PATH=os.path.join(os.getcwd(),'statics','model.h5')
MM_SACLER_SAVE_PATH=os.path.join(os.getcwd(),'statics','mm_scaler.pkl')

# load model & scaler
mm=pickle.load(open(MM_SACLER_SAVE_PATH,'rb'))
model=pickle.load(open(MODEL_SAVE_PATH,'rb'))

# Dict for switch case
heartattack_chance={0:'Less chance',1:'More chance'}

# test case with provided data 

test_df=pd.DataFrame(([65, 1, 3, 142, 220, 1, 0, 158, 0, 2.3, 1, 0, 1, 1],
           [61, 1, 0, 140, 207, 0, 0, 138, 1, 1.9, 2, 1, 3, 0],
           [45, 0, 1, 128, 204, 0, 0, 172, 0, 1.4, 2, 0, 2, 1],
           [45, 0, 1, 128, 204, 0, 0, 172, 0, 1.4, 2, 0, 2, 1],
           [40, 0, 1, 125, 307, 0, 1, 162, 0, 0, 2, 0, 2, 1],
           [48, 1, 2, 132, 254, 0, 1, 180, 0, 0, 2, 0, 2, 1],
           [41, 1, 0, 108, 165, 0, 0, 115, 1, 2, 1, 0, 3, 0],
           [36, 0, 2, 121, 214, 0, 1, 168, 0, 0, 2, 0, 2, 1],
           [45, 1, 0, 111, 198, 0, 0, 176, 0, 0, 2, 1, 2, 0],
           [57, 1, 0, 155, 271, 0, 0, 112, 1, 0.8, 2, 0, 3, 0],
           [69, 1, 2, 179, 273, 1, 0, 151, 1, 1.6, 1, 0, 3, 0]))
test_data_scaled=mm.transform(test_df.iloc[:,0:-1])
outcome=model.predict(test_data_scaled)
print(outcome)
print(accuracy_score(test_df.iloc[:,-1],outcome)*100)

#%% Build the apps using streamlit

with st.form('Heart Attack Prediction Form'):
    st.write("Insert The Patient's info")
    age=int(st.number_input('Age'))
    sex=st.selectbox('Sex',('0', '1'))
    cpt=int(st.selectbox('Chest Pain Type',('0', '1','2','3')))
    trtbps=st.number_input('Resting blood pressure (mm Hg)')
    chol=st.number_input('Serum cholestoral (mg/dl)')
    fbs=st.selectbox('Fasting blood sugar less than 120 mg/dl',('0', '1'))
    restecg=int(st.selectbox('Resting electrocardiographic results',('0', '1','2')))
    thalachh=st.number_input('Maximum heart rate achieved')
    exng=st.selectbox('Exercise induced angina',('0', '1'))
    oldpeak=st.number_input('ST depression')
    slp=st.selectbox('Slope of the peak exercise ST segment',('1', '2','3'))
    caa=st.selectbox('Number of major vessels colored by flourosopy',('0','1', '2','3'))
    thall=st.selectbox('Thalassemia',('0','1', '2','3'))

    submitted=st.form_submit_button('Submit')   # submit button
    
    if submitted==True:
        patient_info=np.array([age,sex,cpt,trtbps,chol,fbs,restecg,
                               thalachh,exng,oldpeak,slp,caa,thall])
                               
        patient_info_scaled=mm.transform(np.expand_dims(patient_info,axis=0))
        outcome=model.predict(patient_info_scaled)

        st.write(heartattack_chance[np.argmax(outcome)])
        
        if np.argmax(outcome)==1:
            st.warning('You are going to get heart attack! Please take care of your health!')
        else:
            st.success('YEAHHH, you are less chance from heart attack!')











