from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd

import warnings
import uvicorn
import traceback
import requests

from database import *
from json_convertor import *


class Request_get(BaseModel):
    Date: str
    Table: str
    Tags: Optional[dict]={}

class Request_send(BaseModel):
    DF: str
    Table: str
    Tags: Optional[list]=[]

class Request_test(BaseModel):
    Date: str


def convert2json(df):
    out=df.copy()
    if df.shape[0]>0:
        out.index=out.index.tz_localize(None)
        out.reset_index(inplace=True)#.to_json()
    return out.to_json()

def conver2df(temp,time_zone_ = None):
    if time_zone_ == None:    
        time_zone_ = 'Etc/GMT-3'
    df=pd.read_json(temp)
    if df.shape[0]>0:
        df['index']=pd.to_datetime(df['index']*1000000)
        df=df.set_index('index')
        df.index=df.index.tz_localize(time_zone_)
    else:
        print('Результат: пустая таблица! Проверьте запрос.')
    return df

def parse_results(out):
    #print(out.text)
    status=out.json()[0]
    
    if status==True:
        dfjson=out.json()[1]
        res=conver2df(dfjson)
    else:
        print('Error in service code:',status)
        res=pd.DataFrame({})
    return res
    
    
app = FastAPI()
# Получение данных из БД    
@app.post("/get")
async def get_(Data: Request_get):
    try:
        overall_status = True
        print('post get. Data:',Data)
        Table=Data.Table
        Date=Data.Date
        Tags=Data.Tags
        res=read_DF_from_influxDB(table_ = Table,timestamp_ = Date, tags_=Tags)
        generic_responce=convert2json(res)
    except:
        overall_status = traceback.format_exc()
        print(overall_status)
        generic_responce={}
    return (overall_status, generic_responce)
    
    
# Передача данных для записи в БД    
@app.post("/send")
async def send_(Data:  Request_send):#: Request_send):
    try:
        overall_status = True
        print('post get. Data:',Data)
        Table=Data.Table
        DF=conver2df(Data.DF)
        Tags=Data.Tags    
        save_df_2_db(DF,table_='Res',Tag_Names=['Ni','Fleet', 'nBoilers','TypeCalc'])
    except:
        overall_status = traceback.format_exc()
        print(overall_status)
        generic_responce={}
    return True
    
@app.post("/test")
async def test(Data: Request_test):
    # получение данных
    print('post test. Data:',Data)
    return pd.DataFrame({'Test':['Сервис запущен!']}).to_json()
    
if __name__ == "__main__":
    #uvicorn.run(app, host="0.0.0.0", port=8005)    
    uvicorn.run(app, host="10.251.0.106", port=8010)   