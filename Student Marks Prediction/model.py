import pandas as pd
import pickle

df = pd.read_csv('student_info.csv')

df2 = df.fillna(df.mean())

X = df2.iloc[:,:1]
y = df2['student_marks']

from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,y,test_size= 0.2,random_state=42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)

file = open('marks_predictor.pkl', 'wb')
pickle.dump(lr, file) 