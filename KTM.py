# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 17:09:15 2020

@author: Gaurav Verma
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv('ktm.csv')
df.head()

le = LabelEncoder()
df.Response = le.fit_transform(df.Response.values)

x = df.iloc[:,1:-1]
y = df.iloc[:,-1]

def gender_to_int(word):
    word_dict = {'Male':1,'Female':2,'Male ':1}
    return word_dict[word]

def occu_to_int(word):
    word_dict = {'Student':1,'Self Employed':2,'Unemployed':3,'Professional':4}
    return word_dict[word]
    
def phone_to_int(word):
    word_dict = {'Low End':0,'Average':1,'High End':2}
    return word_dict[word]
    
def bike_to_int(word):
    word_dict = {'No Bike':0,'Below 125':1,'125 to 180':2,'180 to 220':3,'220 and Above':4}
    return word_dict[word]

def relationship_to_int(word):
    word_dict = {'Single':0,'Committed':1,'Complicated':2,'Married':3}
    return word_dict[word]


x['Gender'] = x['Gender'].apply(lambda x:gender_to_int(x))
x['Occupation'] = x['Occupation'].apply(lambda x:occu_to_int(x))
x['Phone Type'] = x['Phone Type'].apply(lambda x:phone_to_int(x))
x['Current Bike'] = x['Current Bike'].apply(lambda x:bike_to_int(x))
x['Relationship'] = x['Relationship'].apply(lambda x:relationship_to_int(x))

reg = LogisticRegression()
reg.fit(x,y)

pickle.dump(reg, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict_proba([[18,2,1,2,1,2]]))
