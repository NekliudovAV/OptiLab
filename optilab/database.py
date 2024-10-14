import pandas as pd
from pymongo import MongoClient
from influxdb import DataFrameClient

import config
# Пример содержания config.py
# config={'MONGO':{'DB_name':'TES', 'IP_':'127.0.0.1', 'port_' : 27017, 'username_':'mongo', 'password_':'mongo'},
#         'INFLUX':{'DB_name':'TES', 'IP_':'127.0.0.1', 'port_' : 8086}}


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
        db = client[config.MONGO['DB_name']]
        posts = db.posts
        result = posts.insert_many([dict2mongo])
        client.close()

def read_FD_from_mongo(Equipment='T3',Type='DFSt',IP=config.MONGO['IP_']):

        client = MongoClient(IP, config.MONGO['port_'],
                                  username=config.MONGO['username_'],
                                  password=config.MONGO['password_'])
        db = clientconfig.MONGO['DB_name']
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
def write_DF_2_influxDB(resdf, calc_type,  database_,  time_zone_ = None, tags_=None):
    influxDataFrameClient_client = DataFrameClient(host=config.INFLUX['IP_'], port=config.INFLUX['port_'], database=config.INFLUX['DB_name'])
    influx_DBname = calc_type
    influxDataFrameClient_client.write_points(resdf.astype(float), influx_DBname, tags=tags_, batch_size=1000)
    influxDataFrameClient_client.close()
    return True

def read_DF_from_influxDB(host_ = None,
                          port_ = None,
                          database_ = None,
                          table_ = None,
                          timestamp_ = None,
                          time_zone_ = None):
    """
    Запрос из БД InfluxDB предрасчетный параметров 
    Возвращает dataframe с предрасчетными параметрами
    """
    #t0 = time.time()
    if host_==None:
        host_=config.INFLUX['IP_']
    if port_==None:    
        port_=config.INFLUX['port_']
    if database_==None:    
        database_ = config.INFLUX['DB_name']
    if time_zone_ == None:    
        time_zone_ = 'Etc/GMT-3'
    #print('port_',port_, type(port_))    
    influxDataFrameClient_client = DataFrameClient(host = host_, port = port_, database = database_)
    influx_DBname = table_
    df = influxDataFrameClient_client.query(f"""select * from {influx_DBname} where time = '{timestamp_}' tz('{time_zone_}')""")[influx_DBname]    
    df = df.tz_convert(time_zone_)
    influxDataFrameClient_client.close()
    # dt = time.time() - t0
    #print(f'Запрос данных по {df.shape[1]} каналам измерений за {df.shape[0]} периодов выполнен за {dt: 3.3f} c')
    return df

