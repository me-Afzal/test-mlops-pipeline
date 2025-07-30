from fastapi import FastAPI
import pickle
from pydantic import BaseModel

with open('models/model.pkl', 'rb') as f:
    model=pickle.load(f)
    
class inp_data (BaseModel):
    features: list
    
app=FastAPI()

@app.post('/predict')
def predict(data:inp_data):
    prediction=model.predict([data.features])
    return {'prediction':int(prediction[0])}        