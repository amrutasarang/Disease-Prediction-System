import pandas as pd
import pickle
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("book.csv")
df_copy = df.copy(deep=True)

labelencoder_Y = LabelEncoder()
df.iloc[:,1]=labelencoder_Y.fit_transform(df.iloc[:,1].values)

X=df.iloc[:,2:12].valuesS
Y=df.iloc[:,1].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
# print(X[0])
X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)


log=LogisticRegression(random_state=0)
log.fit(X_train,Y_train)
# model=models(X_train,Y_train)

# pred=model.predict(X_test)
# print('Predicted values:')
# print(pred)  
# print('Actual values:')
# print(Y_test)
pickle.dump(log, open('cancer.pkl','wb'))
model = pickle.load(open('cancer.pkl','rb'))

data = np.array([[14.36,566.3,0.06664,23.56,0.02387,0.0023,0.144,0.239,0.2977,0.7259]])
# data = np.array([[15.71,520,0.04568,14.67,0.01698,0.002425,0.1312,0.189,0.3184]])

# data = np.array([[21.38,904.6,0.1525,102.6,0.02741,0.002801,0.1805,0.4695,0.3613,0.09564]])
output = log.predict(data)
print(output)