import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
import sys

df=pd.read_csv('data/processed/IRIS_processed')

x=df.drop(columns=['species'])
y=df.species
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

with open('models/model.pkl', 'rb') as f:
    model=pickle.load(f)
    
y_pred=model.predict(x_test)

acc=accuracy_score(y_test,y_pred)

if acc<0.95:
    print('Model accuracy doesnot meat the criteria')
    sys.exit(1)    