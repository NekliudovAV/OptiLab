import pandas as pd
import time
from pymongo import MongoClient
from influxdb import DataFrameClient

import config
# Пример содержания config.py
# MONGO={'DB_name':'TES', 'IP_':'127.0.0.1', 'port_' : 27017, 'username_':'mongo', 'password_':'mongo'}
# INFLUX={'DB_name':'TES', 'IP_':'127.0.0.1', 'port_' : 8086}


# mongo
# 1. Записать датафрейм в монго
# 2. Прочесть датафрейм из монго
# 3. Получить список таблиц из монго

def mongo_db(IP=config.MONGO['IP_']):
        client = MongoClient(IP, config.MONGO['port_'],
                             username=config.MONGO['username_'],
                              password=config.MONGO['password_'])
        db = client[config.MONGO['DB_name']]
        return db,client 

def write_FD_2mongo(DFStages,Equipment='T3',Type=None,IP=config.MONGO['IP_']):
        if not isinstance(DFStages,pd.DataFrame):
            for k in DFStages.keys():
                if isinstance(DFStages[k],pd.DataFrame):
                    DFStages[k]=DFStages[k].to_json()    
        else:    
                DFStages=DFStages.to_json()
            
        if Type is None:
            EquipmentName=Equipment
        else:    
            EquipmentName=Equipment+'.'+Type
    
        dict2mongo = {'name':EquipmentName,
                      Type : DFStages}

        db,client=mongo_db(IP=IP)
        posts = db.posts
        result = posts.insert_many([dict2mongo])
        client.close()

def read_FD_from_mongo(Equipment='T3',Type=None,IP=config.MONGO['IP_']):
        db,client=mongo_db(IP=IP)
        posts = db.posts
    
        if Type is None:
            EquipmentName=Equipment
        else:    
            EquipmentName=Equipment+'.'+Type
            
        result = list(posts.find({"name":EquipmentName}))[-1]
        client.close()
        print(result)
        
        DFStages={}
        if isinstance(result[Type],dict):
            for key in result[Type].keys():
                DFStages.update({key:pd.read_json(result[Type][key])})
        else:
            DFStages=pd.read_json(result[Type])
        return DFStages  

def list_database_names():
    client = MongoClient(config.MONGO['IP_'], config.MONGO['port_'],
                                  username=config.MONGO['username_'],
                                  password=config.MONGO['password_'])
    out=client.list_database_names(session=None, comment=None)
    client.close()
    return out



# influx
# 1. Создание таблицы
# 2. Записать в таблицу (по умолчанию без тегов)
# 3. Записать в таблицу результаты экспериментов (с тегами)

# Создание новой БД
def add_db(database='KEM_GRES'):
        from   influxdb import InfluxDBClient
        client = InfluxDBClient(host=config.INFLUX['IP_'], port=config.INFLUX['port_'])
        db=client.get_list_database()
        print(db)
        if  database not in  [d['name'] for d in db]:
            client.create_database(database)
        else:
            print('Указанная БД уже существует!')


def write_DF_2_influxDB(resdf, table_=None,  database_ =None,  time_zone_ = None, tags_=None):
    if database_ ==None:
            database_=config.INFLUX['DB_name']
   
    influxDataFrameClient_client = DataFrameClient(host=config.INFLUX['IP_'], port=config.INFLUX['port_'], database=database_)
    influx_DBname = table_
    resdf1=resdf[list(set(resdf.keys())-set(['TimeWrite2DB']))].astype(float)
    if 'TimeWrite2DB' in resdf.keys():
        resdf1['TimeWrite2DB']=resdf['TimeWrite2DB']
    influxDataFrameClient_client.write_points(resdf1, influx_DBname, tags=tags_, batch_size=1000)
    influxDataFrameClient_client.close()
    return True
        
def save_df_2_db(res2,table_='Optimize',database_=None,Tag_Names=['Ni','Fleet', 'nBoilers']):
    if database_ ==None:
            database_=config.INFLUX['DB_name']
            
    Others=list(set(res2.keys())-set(Tag_Names))
    temp=res2[Tag_Names].drop_duplicates()
    print('Уникальные теги:',temp, 'количество уникальных сочетаний:', temp.shape[0])
    for o in range(temp.shape[0]): 
                tt=temp.iloc[o]
                print('Индекс уникального сочетания:',o)
                print(tt)
                # Формируем значения resdf для тега tags_
                for i in range(tt.shape[0]):
                    tags_={}
                    k=0
                    for t in tt.keys():
                        tags_[t]=str(tt[t])
                        if k==0:
                            temp=res2[t]==tt[t]
                            k=k+1
                        else:
                            temp=temp&(res2[t]==tt[t])
                #print(tags_)
                resdf=res2[Others][temp]    
                #print(resdf) # Для отладки
                write_DF_2_influxDB(resdf,table_, database_,tags_=tags_)

def read_DF_from_influxDB(host_ = None,
                          port_ = None,
                          database_ = None,
                          table_ = None,
                          timestamp_ = None,
                          timestamp_to = None,
                          time_zone_ = None,
                          tags_ = None):
    """
    Запрос из БД InfluxDB предрасчетный параметров 
    Возвращает dataframe с предрасчетными параметрами
    """
    t0=time.time()
    timestamp_=pd.Timestamp(timestamp_)
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
    
    if timestamp_to == None:
        timestamp_to=pd.Timestamp(timestamp_)
    else:
        timestamp_to=pd.Timestamp(timestamp_to)
    tags_c=''
    if not tags_==None:
        for k in tags_.keys():
            tags_c=tags_c+(f" {k}='{str(tags_[k])}' and")
        
    query=f"""select * from {table_} where {tags_c}  time >= '{timestamp_}'  and time <= '{timestamp_to}' tz('{time_zone_}')"""
    print(query)
    df = influxDataFrameClient_client.query(query)
    if table_ in df.keys():
        df=df[table_]        
        df = df.tz_convert(time_zone_)
    else:
        print('Результат запроса - пустая таблица')
        df =pd.DataFrame()
    
    influxDataFrameClient_client.close()
    dt = time.time() - t0
    print(f'Запрос на получение данных из {table_} c {timestamp_} по {timestamp_to}  выполнен за {dt: 3.3f} c')
    return df

