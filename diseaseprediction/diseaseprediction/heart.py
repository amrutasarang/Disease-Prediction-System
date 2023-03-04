import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

df=pd.read_csv('heart.csv')

X=df.drop(columns='target',axis=1)
y=df['target']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=1)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression().fit(X_train,y_train)
y_pred=model.predict(X_test)
print('Accurary of model is: ',model.score(X_test,y_test)*100)



pickle.dump(model, open('model1.pkl','wb'))
m = pickle.load(open('model1.pkl','rb'))

# data = np.array([[60,1,0,117,230,1,1,160,1,1.4,2,2,3]])
data = np.array([[52,1,0,125,212,0,1,168,0,1.0,2,2,3]])

output = model.predict(data)
print(output)