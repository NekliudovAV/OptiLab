import pandas as pd
import time
from pymongo import MongoClient
from influxdb import DataFrameClient,InfluxDBClient
from json_convertor import *

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
        


def write_DF_2mongo(DFSt2,Equipment='TA3',Name='D0',Subsystem='St2',Model='Base',IP=None):
        # 
        if IP==None:
            IP=config.MONGO['IP_']
            
        DFSt2=convert2jsonMongo(DFSt2)
        
        #if isinstance(DFSt2,pd.DataFrame):
        #    DFSt2=DFSt2.to_json()
        #else:    
        #    for k in DFSt2.keys():
        #        if isinstance(DFSt2[k],pd.DataFrame):
        #            DFSt2[k]=DFSt2[k].to_json()
        
        name=Equipment            
        if not Subsystem==None:
            name=name+'.'+Subsystem
        name=name+'.'+Name    
        if not Model=='Base':
            name=name+':'+Model
            
        dict2mongo = {'name':name,
                      'Equipment':Equipment,
                      'Subsystem': Subsystem,
                      'Name':Name,
                      'Model':Model,
                      'Type': 'Curve',
                      'DF' : DFSt2}

        client = MongoClient(IP, config.MONGO['port_'],
                             username=config.MONGO['username_'],
                              password=config.MONGO['password_'])
        DB_Name=config.MONGO['DB_name']
        db = client[DB_Name]
        posts = db.posts
        result = posts.insert_many([dict2mongo])
        client.close()

def delete_from_mongo_by_name(name='TA3.St2.Qt:Test'):
    db,client=mongo_db()
    query={'name':name} 
    posts = db.posts
    result=posts.delete_many(query)
    client.close()
    print(f"Deleted {result.deleted_count} document(s).")
    return True

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
        
        DFStages=onvertMongoJson2DF(result[Type])
        
        #DFStages={}
        #if isinstance(result[Type],dict):
        #    for key in result[Type].keys():
        #        DFStages.update({key:pd.read_json(result[Type][key])})
        #else:
        #    DFStages=pd.read_json(result[Type])
        return DFStages  

def list_database_names():
    client = MongoClient(config.MONGO['IP_'], config.MONGO['port_'],
                                  username=config.MONGO['username_'],
                                  password=config.MONGO['password_'])
    out=client.list_database_names(session=None, comment=None)
    client.close()
    return out
    
def get_list(Tags=None):
    # Список всех записей
        client = MongoClient(config.MONGO['IP_'], config.MONGO['port_'],
                                  username=config.MONGO['username_'],
                                  password=config.MONGO['password_'])
        #db = client.KemGRES
        DB_Name=config.MONGO['DB_name']
        db = client[DB_Name]
        
        posts = db.posts
        if Tags ==None:
            query = {}
        else:
            query = Tags
        projection = {"_id":1,"name":1,"Equipment":1,"Subsystem":1,"Name":1,"Model":1,"Type":1}
        result = list(posts.find(query,projection))
        
        client.close()
        dict_for_df=[]
        for i in result:
            temp=pd.DataFrame({k:[i[k]] for k in i.keys()})
            dict_for_df.append(temp)
        res=pd.concat(dict_for_df).reset_index(drop=True) 
        if 'Equipment' in res.keys():
            res=res.iloc[res[['name','Equipment']].drop_duplicates().index].set_index('_id')
        else:
            res=res.iloc[res[['name']].drop_duplicates().index].set_index('_id')        
        return res
        
def get_DF(Name='TA3.DFSt2',df=None):
    client = MongoClient(config.MONGO['IP_'], config.MONGO['port_'],
                                  username=config.MONGO['username_'],
                                  password=config.MONGO['password_'])
    db = client[config.MONGO['DB_name']]
    posts = db.posts
    
    if isinstance(df,pd.DataFrame):
        ID_=df.loc[Name][0]    
        query = {"_id":ID_}
    else:
        query = {"name":Name}
    projection = {"_id":0,"name":0}
    result = list(posts.find(query,projection))
    client.close()
    result=result[-1]
    
    if 'DF' in result.keys(): 
        DFSt2_=pd.read_json(result['DF'])
    else:
        Type=list(result.keys())[0]    
        DFSt2_={}
        for key in result[Type].keys():
            DFSt2_.update({key:pd.read_json(result[Type][key])})
    return DFSt2_
    

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
            
def drop_measurement(Name):
    client = InfluxDBClient(host=config.INFLUX['IP_'], port=config.INFLUX['port_'],database=config.INFLUX['DB_name'])
    client.drop_measurement(Name)
    client.close()
    

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
        
def save_df2influx(df,Table='basic',Station='KemGRES',Equipment='All',TypeCalc="calc", Scenario="Base",Model="Base",Version='1'):
    Tag_Names=['Station','Equipment','TypeCalc','Scenario','Model','Version']
    df_keys=df.keys()
    if 'Station' not in df_keys:
        df['Station']=Station
    if 'Equipment' not in df_keys:
        df['Equipment']=Equipment
    if 'TypeCalc' not in df_keys:
        df['TypeCalc']=TypeCalc
    if 'Scenario' not in df_keys:
        df['Scenario']=Scenario
    if 'Model' not in df_keys:
        df['Model']=Model
    if 'Version' not in df_keys:
        df['Version']=Version
    save_df_2_db(df,table_=Table,database_=None,Tag_Names=Tag_Names)
        
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
                            ftemp=res2[t]==tt[t]
                            k=k+1
                        else:
                            ftemp=ftemp&(res2[t]==tt[t])
                #print(tags_)
                resdf=res2[Others][ftemp]    
                #print(resdf) # Для отладки
                write_DF_2_influxDB(resdf,table_, database_,tags_=tags_)
                
def read_influx(date,Table='basic',Station='KemGRES',date_to=None,Equipment='All',TypeCalc="calc", Scenario="Base",Model="Base",Version='1',database_=None,time_zone_=None,host_=None):
    Tags={}
    if not Station==None:
        Tags['Station']=Station
    if not Equipment==None:
        Tags['Equipment']=Equipment
    if not TypeCalc==None:        
        Tags['TypeCalc']=TypeCalc
    if not Scenario==None:    
        Tags['Scenario']=Scenario
    if not Model==None:    
        Tags['Model']=Model
    if not Version==None:
        Tags['Version']=Version
    if date_to==None:
        date_to=date
    
    return read_DF_from_influxDB(table_=Table,timestamp_=date,timestamp_to=date_to,tags_=Tags,
                                 database_=database_,time_zone_=time_zone_,host_=host_)
    

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
    query=f"""select * from {table_} where {tags_c}  time >= '{timestamp_}'  and time <= '{timestamp_to}' """
    if len(time_zone_)>0:
        query=query+f""" tz('{time_zone_}')"""
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

def read_DF_from_influxDB_unstack(host_ = None,
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
    out_=read_DF_from_influxDB(host_ = host_,port_ = port_,database_ = database_,table_ = table_,timestamp_ = timestamp_,timestamp_to = timestamp_to,time_zone_ = time_zone_,tags_ = tags_)
    if 'value' in out_.keys() and 'name' in out_.keys():
        out_=out_[['name','value']].reset_index().set_index(['index','name']).unstack()
        out_.columns=out_.columns.droplevel()
    return out_
