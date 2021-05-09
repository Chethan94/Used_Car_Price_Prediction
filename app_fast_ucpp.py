# https://blog.finxter.com/deploying-a-machine-learning-model-in-fastapi/
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle

app = FastAPI()


pickle_in = open("used_car_price_prediction.pkl","rb")
model=pickle.load(pickle_in)

@app.get("/")
async def root():
    return {"message": "Model is working He He"}

@app.post("/predict")
def predict(car_condition:float,age: int, transmission: str,insurance: str,fuel_type:str):
    if(insurance=='yes'):
        ins=1
    else:
        ins=0
    if(transmission=='manual'):
        tran=1
    else:
        tran=0
    if(fuel_type=='petrol'):
        a=1
        b=0
        c=0
    elif(fuel_type=='petrol+cng'):
        a=0
        b=1
        c=0
    elif(fuel_type=='petrol+lpg'):
        a=0
        b=0
        c=1
    else:
        a=0
        b=0
        c=0
    data = np.array([1.0,car_condition,age,a,b,c,tran,ins,car_condition*age])
    prediction = model.predict(data)
    return {
        'prediction': prediction[0],
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
