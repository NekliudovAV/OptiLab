import pandas as pd
def convert2json(df):
    out=df.copy()
    if df.shape[0]>0:
        out.index=out.index.tz_localize(None)
        out.reset_index(inplace=True)#.to_json()
    return out.to_json()

def convert2df(temp,time_zone_ = None):
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