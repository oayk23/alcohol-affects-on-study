# Importing Necessary librarys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import layers
from keras import models
# Reading Data
data2 = pd.read_csv('Maths.csv')
data1 = (pd.read_csv("Maths.csv"))

#Encoding
le = LabelEncoder()

data1['school'] = le.fit_transform(data1['school'])
data1['sex'] = le.fit_transform(data1['sex'])
data1['address'] = le.fit_transform(data1['address'])
data1['Pstatus'] = le.fit_transform(data1['Pstatus'])
data1['schoolsup'] = le.fit_transform(data1['schoolsup'])
data1['famsup'] = le.fit_transform(data1['famsup'])
data1['paid'] = le.fit_transform(data1['paid'])
data1['activities'] = le.fit_transform(data1['activities'])
data1['nursery'] = le.fit_transform(data1['nursery'])
data1['higher'] = le.fit_transform(data1['higher'])
data1['internet'] = le.fit_transform(data1['internet'])
data1['romantic'] = le.fit_transform(data1['romantic'])
data1['guardian'] = le.fit_transform(data1['guardian'])
data1['famsize'] = le.fit_transform(data1['famsize'])
data1['Mjob'] = le.fit_transform(data1['Mjob'])
data1['Fjob'] = le.fit_transform(data1['Fjob'])
data1['reason'] = le.fit_transform(data1['reason'])
#train-test splitting
x = data1.iloc[:,0:30].values
y = data1.iloc[:,30:].values
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75,random_state=0)
#Scaling
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Creating our ANN Model
seq = models.Sequential()
seq.add(layers.Dense(units=30,activation='relu',input_shape=(X_train[0].shape)))
seq.add(layers.Dense(units=6,activation='relu'))
seq.add(layers.Dense(units=3,activation='sigmoid'))
#compiling ANN Model
seq.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
seq.fit(X_train,y_train,batch_size=32,epochs=13)

#Predict Session
y_pred = seq.predict(X_test)

#Result
seq_list = pd.DataFrame(list(zip(y_test, y_pred)), columns=['Actual', 'Predicted'])







































