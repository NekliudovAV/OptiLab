import requests
import pandas as pd
import time
import json

from json_convertor import *

# Для роаботы класса должен быть запущен сервис InfuxAPI
# Класс для удобства передачи данных

def parse_results(out):
        #print(out.text)
        status=out.json()[0]

        if status==True:
            dfjson=out.json()[1]
            res=convert2df(dfjson)
        else:
            print('Error in service code:',status)
            res=pd.DataFrame({})
        return res

def correct_tags(df):
    df=df.rename(columns={'fleet':'Fleet','n_boilers':'nBoilers'})
    return df

class Influx_API(object):
    def __init__(self,host=None):
        if host==None:
            host='10.251.0.106:8010'
        self.host=host
        
    #@staticmethod    

    def get_calc(self,Date):
        data2send = {'Date': Date,'Table':'calc'}
        out = requests.post(f"http://{self.host}/get",json=data2send)
        return parse_results(out)

    def get(self,Date,Table,Tags={}):
        data2send = {'Date': Date,'Table':Table, 'Tags':Tags}
        out = requests.post(f"http://{self.host}/get",json=data2send)
        return parse_results(out)

    def send(self,DF,Table,Tags=[]):
        data2send = {'DF': convert2json(DF),'Table':Table, 'Tags':Tags}
        out = requests.post(f"http://{self.host}/send",json=data2send)
        print(out.text)
        return out
        
    def test(self):
        data2send = {'Date': 'test'}
        res = requests.post(f"http://{self.host}/test",json=data2send)
        return pd.read_json(res.json())

    # Получение данных
    # IAPI=Influx_API()
    # out=IAPI.get('2025.02.19 00:00','BRAC2',Tags={'Ni':'N0','n_boilers':'6.0'})
    # Запись
    # out=send(out1,'Res',Tags=['Ni','Fleet', 'nBoilers','TypeCalc'])#'Ni':'plan',
    # Проверка записанных данных
    # get('2025.02.19 00:00','Res',Tags={'TypeCalc':'BRAC'})

    # Далее функции, которые упрощают передачу данных:
    def send_use(self,Use,Tags=['TypeCalc','Scenario']):
        if not 'Scenario' in Use.keys():
            Use['Scenario']='Base'
        if not 'TypeCalc' in Use.keys():
            Use['TypeCalc']='Fact'

        out=send(self,Use,'Use',Tags=['TypeCalc','Scenario'])
        return out

    def send_limits(self,Limits,Tags=['TypeLimits','TypeCalc','Scenario']) :
        if not 'Scenario' in Limits.keys():
            Limits['Scenario']='Nominal'
        if not 'TypeCalc' in Limits.keys():
            Limits['TypeCalc']='All'
        if not 'TypeCalc' in Limits.keys():
            Limits['TypeLimits']='Max'

        out=send(Limits,'Limts',Tags=['TypeCalc','Scenario','TypeLims'])
        return out

    def send_fixed(self,Fixed,Tags=['TypeCalc','Scenario']) :
        if not 'Scenario' in Fixed.keys():
            Fixed['Scenario']='Nominal'
        if not 'TypeCalc' in Fixed.keys():
            Fixed['TypeCalc']='Fact'

        out=send(Fixed,'Limts',Tags=['TypeCalc','Scenario'])
        return out