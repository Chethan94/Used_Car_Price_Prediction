# https://blog.finxter.com/deploying-a-machine-learning-model-in-fastapi/
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
origins = ["*"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



pickle_in = open("used_car_price_prediction.pkl","rb")
model=pickle.load(pickle_in)

@app.get("/")
async def root():
    return {"message": "Model is working He He"}

@app.get("/predict")
async def predict(car_condition:float,age: int, transmission: str,insurance: str,fuel_type:str,owner:int,brand:int,km:int):
    xsp=420129.97839
    ssp=225679.2812696
    xkm=58827.68919
    skm=34652.8969288
    xcc=4.381747
    scc=0.2782
    xa=7.167816
    sa=2.86251566
    xb=10.0827586
    sb=3.5695935
    if(insurance=='yes'):
        ins=1
    else:
        ins=0
    if(owner==1):
        s=0
        t=0
    elif(owner==2):
        s=1
        t=0
    else:
        s=0
        t=1
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
    data = np.array([1.0,((km-xkm)/skm),((car_condition-xcc)/scc),((age-xa)/sa),((brand-xb)/sb),s,t,a,b,c,tran,ins,car_condition*age])
    prediction = model.predict(data)
    pred=abs((prediction[0]*ssp)+xsp)
    return {
        'p': pred,
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
