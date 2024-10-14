import pandas as pd
from pymongo import MongoClient
from influxdb import DataFrameClient

import config
# Пример содержания config.py
# config={'MONGO':{'mongoDB_name':'biysk', 'IP_':'127.0.0.1', 'port_' : 27017, 'username_':'mongo', 'password_':'mongo','database_':'TES'},
#         'INFLUX':{'InfluxDB_name':'biysk2', 'IP_':'127.0.0.1', 'port_' : 8086}}


# mongo
def write_FD_2mongo(DFStages,Equipment='T3',Type='DFSt',IP=config.MONGO['IP_']):
        for k in DFStages.keys():
            if isinstance(DFStages[k],pd.DataFrame):
                DFStages[k]=DFStages[k].to_json()
        dict2mongo = {'name':Equipment+'.'+Type,
                      Type : DFStages}

        client = MongoClient(IP, config.MONGO['port_'],
                             username=config.MONGO['username_'],
                              password=config.MONGO['password_'])
        db = client[config.MONGO['TES']]
        posts = db.posts
        result = posts.insert_many([dict2mongo])
        client.close()

def read_FD_from_mongo(Equipment='T3',Type='DFSt',IP=config.MONGO['IP_']):

        client = MongoClient(IP, config.MONGO['port_'],
                                  username=config.MONGO['username_'],
                                  password=config.MONGO['password_'])
        db = clientconfig.MONGO['TES']
        posts = db.posts
        result = list(posts.find({"name":Equipment+'.'+Type}))[-1]
        client.close()
        DFStages={}
        for key in result[Type].keys():
            DFStages.update({key:pd.read_json(result[Type][key])})
        return DFStages  

def list_database_names():
    client = MongoClient(config.MONGO['IP_'], config.MONGO['port_'],
                                  username=config.MONGO['username_'],
                                  password=config.MONGO['password_'])
    out=client.list_database_names(session=None, comment=None)
    client.close()
    return out

# def add_database():
# creates databases and collections automatically for you if they don't exist already. 

# influx
