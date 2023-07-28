
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('zomato_df.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)
print(df.head())
x=df.drop('rate',axis=1)
y=df['rate']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=10)



from sklearn.ensemble import  RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(x_train,y_train)


y_predict=rf.predict(x_test)


import pickle
pickle.dump(rf, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(y_predict)