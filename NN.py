import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from sklearn.metrics import accuracy_score
from keras.layers import Dense
from numpy import array
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
from keras.models import load_model
dataset = pd.read_csv(r'C:\Users\saikr\Desktop\Covid Dataset.csv')
print(dataset.head(3))
number=LabelEncoder()
for col in dataset.columns:
    dataset[col]=number.fit_transform(dataset[col].astype('str'))
print(dataset.head(3))
dataset['kidney disease']=dataset['Gastrointestinal']
dataset['travel-history']=dataset['Abroad travel']+dataset['Contact with COVID Patient']+dataset['Attended Large Gathering']+dataset['Visited Public Exposed Places']
dataset['safety']=dataset['Wearing Masks']+dataset['Sanitization from Market']
i=2
for i in range(5):
    dataset['travel-history'].replace(to_replace=i,value=1,inplace=True)
dataset['safety'].replace(to_replace=1,value=2,inplace=True)
dataset['travel-history']=dataset['travel-history']-dataset['safety']
j=-2
for j in range(0):
    dataset['travel-history'].replace(to_replace=j,value=0,inplace=True)
dataset['lung disease']=dataset['Asthma']+dataset['Chronic Lung Disease']
dataset['lung disease'].replace(to_replace=2,value=1,inplace=True)
dataset['aches-pains']=dataset['Headache']
dataset['Corona result']=dataset['COVID-19']
dataset=dataset.drop(columns=['COVID-19','Family working in Public Exposed Places','Gastrointestinal','safety','Abroad travel','Contact with COVID Patient','Asthma','Chronic Lung Disease','Attended Large Gathering','Visited Public Exposed Places','Wearing Masks','Sanitization from Market','Running Nose','Headache'])
print(dataset['travel-history'].value_counts())
print("upated columns after generalization")
for col in dataset.columns:
    print(col)
d= preprocessing.normalize(dataset, axis=0)
h = pd.DataFrame(d)
X = h.values[:,0:12]
Y=dataset['Corona result']
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3)

model =Sequential()
model.add(Dense(8, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100,batch_size=400)
y_pred = model.predict(X_test)
c=confusion_matrix(y_test,y_pred)
print(c)
print("Accuracy of source:",accuracy_score(y_test, y_pred)*100)
model.save('model.h5')
'''
model =load_model('model.h5')
model.layers[1].trainable=False
model.layers[2].trainable=False
model.layers[3].trainable=False
model.layers[0].trainable=False
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100,batch_size=3335)
y_pred = model.predict(X_test)
print("Accuracy with transfer learning with freezing weight:",accuracy_score(y_test, y_pred)*100)
'''

