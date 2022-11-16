# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:07:45 2022

@author: DELL
"""

import numpy as np
import pandas as pd
import sklearn
import pickle
dataset=pd.read_csv(r"C:\Users\DELL\Downloads\Crop_recommendation (1).csv")
x=dataset.drop('label',axis=1).values
y=dataset['label'].values
# Label Encoding the  column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=0)
from xgboost import XGBClassifier
classifier = XGBClassifier() 
classifier.fit(x_train, y_train)
from xgboost import XGBClassifier
classifier = XGBClassifier() 
classifier.fit(x_train, y_train)
y_pred=classifier.predict(x_test)
new_input=np.array([[20,30,20,20,70,2,203]])
columns=['N',	'P','K','temperature','humidity','ph','rainfall']
prediction_test=classifier.predict(new_input)
list(le.classes_)
list(le.inverse_transform([prediction_test]))
filename='soil_health_crop.sav'
pickle.dump(classifier,open(filename,'wb'))
#loading the model
loaded_model=pickle.load(open("soil_health_crop.sav",'rb'))
input_data=(30,60,10,50,70,6,203)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-2)
predic=loaded_model.predict(input_data_reshaped)
print(predic)
list(le.classes_)
list(le.inverse_transform([predic]))