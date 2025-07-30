import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df=pd.read_csv('data/processed/IRIS_processed')

x=df.drop(columns=['species'])
y=df.species
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(n_estimators=50,random_state=42)
model.fit(x_train,y_train)

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model,f)

