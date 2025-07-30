from app.main import app
from fastapi.testclient import TestClient

client=TestClient(app)

def test_app():
    data={'features':[4.6,3.4,1.4,0.3]}
    response=client.post('/predict')
    
    assert response.status_code == 200
    
    json_data=response.json()
    
    assert 'prediction' in json_data
    
    assert json_data['prediction'] in [0,1,2]