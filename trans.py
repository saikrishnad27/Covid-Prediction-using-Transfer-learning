import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from sklearn.metrics import accuracy_score
from keras.layers import Dense
from sklearn import preprocessing
from numpy import array
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from keras import Input
from keras.models import load_model
from sklearn.metrics import confusion_matrix
#Reading dataset
dataset = pd.read_excel(r'C:\Users\saikr\Desktop\COVID-19.xlsx')
print(dataset.head(3))
dataset.loc[dataset['body temperature']>98.6,'Fever']=1
dataset.loc[dataset['body temperature']<=98.6,'Fever']=0
dataset['Fever']=dataset['Fever'].astype(int)
print(dataset['Fever'].head(3))
dataset['Sore throat']=dataset['sour throat']
dataset['Breathing Problem']=dataset['breathing problem']
dataset['Fatigue']=dataset['drowsiness']
dataset['Heart Disease']=dataset['heart disease']
dataset['Diabetes']=dataset['diabetes']
dataset['Hyper Tension']=dataset['high blood pressue']
dataset['travel-history']=dataset['travel history to infected countries']
dataset['aches-pains']=dataset['Loss of sense of smell']
#dataset['Corona result'].replace(to_replace=1,value=0,inplace=True)
dataset['Corona result'].replace(to_replace=2,value=1,inplace=True)
dataset=dataset.drop(columns=['Sno','body temperature','sour throat','breathing problem','drowsiness','heart disease','diabetes','high blood pressue','Loss of sense of smell','travel history to infected countries','age','gender','weakness','pain in chest','stroke or reduced immunity','symptoms progressed','change in appetide'])
dataset=dataset.reindex(columns=['Breathing Problem','Fever','Dry Cough','Sore throat','Heart Disease','Diabetes','Hyper Tension','Fatigue','kidney disease','travel-history','lung disease','aches-pains','Corona result'])
for col in dataset.columns:
    print(col)
#number=LabelEncoder()
#for col in dataset.columns:
    #dataset[col]=number.fit_transform(dataset[col].astype('str'))
print(dataset.count())
dataset.fillna(dataset.mean(),inplace=True)

h=preprocessing.StandardScaler().fit(dataset.values[0:125,0:12])
ht=h.transform(dataset.values[0:125,0:12])
X=pd.DataFrame(ht)
Y=dataset['Corona result']
Y=Y[0:125]
print(dataset.shape)
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

model =Sequential()
model.add(Dense(8, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100,batch_size=125)
y_pred = model.predict(X_test)
c=confusion_matrix(y_test,y_pred)
print(c)
print("Accuracy of source :",accuracy_score(y_test, y_pred)*100)

model =load_model('model.h5')
model.layers[1].trainable=False
model.layers[2].trainable=False
model.layers[3].trainable=False
model.layers[0].trainable=False
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10,batch_size=15,validation_data=(X_val, y_val))
print(history.history.keys())
loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy with model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred = model.predict(X_test)
c=confusion_matrix(y_test,y_pred)
print(c)
print("Accuracy with transfer learning without freezing layers:",accuracy_score(y_test, y_pred)*100)


