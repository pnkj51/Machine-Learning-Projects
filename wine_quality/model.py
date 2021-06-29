
import pandas as pd 
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('wine_df.csv')

df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]

X = df.drop(['quality','goodquality'], axis = 1)
y=df['quality']

X = StandardScaler().fit_transform(X)

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.3)

booster = GradientBoostingClassifier()
booster.fit(Xtrain,ytrain)

pickle.dump(booster,open('model.pkl','wb'))


