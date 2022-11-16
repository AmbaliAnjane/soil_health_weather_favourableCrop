# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:16:55 2022

@author: DELL
"""

import numpy as np
import pickle
import streamlit as st
import sklearn
from sklearn.preprocessing import LabelEncoder
#loading the saved  model
loaded_model=pickle.load(open("C:/Users/DELL/crop_prediction_forsoil/soil_health_crop.sav",'rb'))
loaded_model=pickle.load(open("soil_health_crop.sav",'rb'))
input_data=(30,60,10,50,70,6,203)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-2)
predic=loaded_model.predict(input_data_reshaped)
print(predic)
file = open("le.obj",'rb')
le_loaded = pickle.load(file)
file.close()
list(le_loaded.classes_)
print(list(le_loaded.inverse_transform([predic])))
#creating a function
def crop_prediction(input_data):
    input_data_as_numpy_array=np.asarray(input_data,dtype=int)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-2)
    predic=loaded_model.predict(input_data_reshaped)
    list(le_loaded.classes_)
    return(list(le_loaded.inverse_transform([predic])))
def main():
    #giving title
    st.title('KNOW THE CROP FAVOURABLE FOR YOUR SOIL')
#columns=['N',	'P','K','temperature','humidity','ph','rainfall']    
    N=st.text_input('nitrogen')
    P=st.text_input('phousphorouse')
    K=st.text_input('potassium')
    temperature=st.text_input('temperature')
    humidity=st.text_input('humidity')
    ph=st.text_input('ph')
    rainfall=st.text_input('rainfall')
    #code for prediction
    conclude=''
    #creating button
    if st.button('my_crop'):
       conclude=crop_prediction([N,P,K,temperature,humidity,ph,rainfall])
    st.success(conclude)  
if __name__=='__main__':
   main()