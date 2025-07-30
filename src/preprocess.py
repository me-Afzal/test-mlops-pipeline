import pandas as pd
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv('data/raw/IRIS.csv')

encoder=LabelEncoder()
df.species=encoder.fit_transform(df.species)

df.to_csv('data/processed/IRIS_processed',index=False)