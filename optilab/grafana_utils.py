import json
import os
import codecs
import pandas as pd
import numpy as np
import string
import copy

# Сохранение таблицы с переменными DrawIO
def Draio2Table(DrawIOFile='TA8.xml',XlsFile='Table_id1.xlsx'):
    tree = ET.parse(DrawIOFile)
    root = tree.getroot()
    diagram = root[0]
    Table=[]
    for mxCell in diagram.iter('mxCell'):
        id=mxCell.get('id')
        if mxCell.get('value') is None:
            Value=''
        else:
            Value=mxCell.get('value')
            Value=Value.replace('&nbsp;','')
            out=re.search('>\w*\s?\w*</',Value)
            if not out is None:
                Value=out.group(0)[1:-2]
        if mxCell.get('style') is None:
            figure=''
        else:
            figure=mxCell.get('style').split(';')[0]
        if len(Value)>0:   
            Table.append(pd.DataFrame({'id':[id],'Переменная':[Value],'Добалвение текста':['anl'],'Цвет':[''],'Тип фигуры':[figure]}))
    Table=pd.concat(Table)
    Table=Table.reset_index().drop(columns=['index'])
    Table.to_excel(XlsFile)
    return  Table
    
def add_var(List=[],datasource="InfluxDB",name="calculation",query="SHOW MEASUREMENTS", label=None):
    null=None
    false=False
    if label is None:
        label=name
    temp= {
        "allValue": null,
        "current": {
          "selected": false,
          "text": "fact",
          "value": "fact"
        },
        "datasource": datasource,
        "definition": query,
        "description": null,
        "error": null,
        "hide": 0,
        "includeAll": false,
        "label":label,
        "multi": false,
        "name": name,
        "options": [],
        "query": query,
        "refresh": 1,
        "regex": "",
        "skipUrlSync": false,
        "sort": 0,
        "type": "query"
      }
    List.append(temp)
    

def templating_list(datasource="InfluxDB",tags=['Fleet','Ni','nboilers']):
    
    VarInDB='T8.D0'
    List=[]
    add_var(List=List,datasource=datasource)
    add_var(List=List,datasource=datasource,name="calculation") # Добавляем таблицы с резкльтатами
    add_var(List=List,datasource=datasource,name=tags[0], query="select temp from (select "+tags[0]+"::tag as temp,"+VarInDB+"  from $calculation where $timeFilter )")
    add_var(List=List,datasource=datasource,name=tags[1], query="select temp from (select "+tags[0]+"::tag,"+tags[1]+"::tag as temp, "+VarInDB+"  from $calculation where $timeFilter)")
    add_var(List=List,datasource=datasource,name=tags[2], query="select temp from (select "+tags[0]+"::tag,"+tags[1]+"::tag,"+tags[2]+"::tag as temp, "+VarInDB+"  from $calculation where $timeFilter )")
    List
    return { "list":List}

def get_example(Type=1):
    example1_={"alias": "TA3_N",
          "groupBy": [
            {
              "params": [
                "$__interval"
              ],
              "type": "time"
            },
            {
              "params": [
                "null"
              ],
              "type": "fill"
            }
          ],
          "orderByTime": "ASC",
          "policy": "default",
          "query": "SELECT mean(\"value\") FROM /^$calculation$/ WHERE (\"name\" = 'TA3.N') AND $timeFilter GROUP BY time($__interval) fill(none)",
          "rawQuery": True,
          "refId": "A",
          "resultFormat": "time_series",
          "select": [
            [
              {
                "params": [
                  "value"
                ],
                "type": "field"
              },
              {
                "params": [],
                "type": "mean"
              }
            ]
          ],
          "tags": [
            {
              "key": "name",
              "operator": "=",
              "value": "TA3.N"
            }
          ]
        }
    return example1_
    
def get_query2(fname='КА11.D0L2',refId='A',table='/^$calculation$/',AddName=''):
    #fname='T8.'+fname
    temp={'alias': fname.replace('.','_'),
     'groupBy': [{'params': ['$__interval'], 'type': 'time'},
      {'params': ['null'], 'type': 'fill'}],
     'orderByTime': 'ASC',
     'policy': 'default',
     'query': 'SELECT mean(\"'+fname+'\") FROM '+ table +' WHERE (\"Ni\"::tag =~ /^$Ni$/ AND \"Fleet\"::tag =~ /^$Fleet$/) AND (\"nboilers\"::tag =~ /^$nboilers$/) AND $timeFilter GROUP BY time($__interval) fill(none)',
     'rawQuery': True,
     'refId': refId,
     'resultFormat': 'time_series',
     'select': [[{'params': ['value'], 'type': 'field'},
        {'params': [], 'type': 'mean'}]],
     'tags': [{'key': 'name', 'operator': '=', 'value': fname}]}
    return temp

# Формирование Json для заполнения цветом и стрелочек
def get_rools(VarName='КА11.D0',Shape='',Text='',Add_Text='anl'):
    aliace=VarName.replace('.','_')
    temp={'aggregation': 'current',
     'alias': aliace,
     'colors': ['#3274D9', '#56A64B', '#FF780A', '#E02F44'],
     'column': 'Time',
     'dateFormat': 'YYYY-MM-DD HH:mm:ss',
     'decimals': 2,
     'eventData': [],
     'eventProp': 'id',
     'eventRegEx': False,
     'gradient': False,
     'hidden': False,
     'invert': False,
     'linkData': [],
     'linkProp': 'id',
     'linkRegEx': True,
     'mappingType': 1,
     'metricType': 'serie',
     'order': 1,
     'overlayIcon': False,
     'pattern': aliace,
     'rangeData': [],
     'reduce': True,
     'refId': 'A',
     'sanitize': False,
     'shapeData': [{'colorOn': 'a',
       'hidden': False,
       'pattern': Shape,
       'style': 'fillColor'}],
     'shapeProp': 'id',
     'shapeRegEx': True,
     'stringThresholds': ['/.*/'],
     'textData': [{'hidden': False,
       'pattern': Text,
       'textOn': 'wmd',
       'textPattern': '/.*/',
       'textReplace': Add_Text}],
     'textProp': 'id',
     'textRegEx': True,
     'thresholds': [0, 20, 100],
     'tooltip': False,
     'tooltipColors': False,
     'tooltipLabel': '',
     'tooltipOn': 'a',
     'tpDirection': 'v',
     'tpGraph': False,
     'tpGraphHigh': None,
     'tpGraphLow': None,
     'tpGraphScale': 'linear',
     'tpGraphSize': '100%',
     'tpGraphType': 'line',
     'type': 'number',
     'unit': 'short',
     'valueData': []}
    return temp

def correct_Gr_Json(JsonFile,DataFile,DrawIOFile,Type=1):
    # Чтение данных модели
    with codecs.open(JsonFile, "r","utf_8_sig") as json_file:
        data_j=json.load(json_file)

    # Чтение шаблона
    import copy
    # Берём тэги из Файла:
    data=pd.read_excel(DataFile)
    data.head()

    # Формирование правил данных и добаление из в json
    out=[]
    out2=[]
    for i in data.index:
        #example1=copy.deepcopy(data_j['panels'][0]['rulesData']['rulesData'][0]);
        #example2=copy.deepcopy(data_j['panels'][0]['rulesData']['rulesData'][0]);
        fname=data['Переменная'][i]
        shape_color=data['Цвет'][i]
        if shape_color is None or shape_color  is np.NaN:
            shape_color=""
        id=data['id'][i]
        add_text=data['Добалвение текста'][i]
        print(i,fname,shape_color,id)
        
        out.append(get_rools(fname,shape_color,id,add_text))
        #out2.append(get_rools(fname,shape_color,text,add_text))

    # Перезапись в Json
    data_j['panels'][0]['rulesData']['rulesData']=out
    #if len(data_j['panels'])>1:
    #    data_j['panels'][1]['rulesData']['rulesData']=out2

    #import copy

    data_t=data[['Переменная']]
    data_t=data_t.rename(columns = {'Переменная':'value'})


    out=[]
    out2=[]
    for i, fname in enumerate(data_t.value):
        #example1=copy.deepcopy(data_j['panels'][0]['targets'][0]);
        example1=get_example()
        #if len(data_j['panels'])>1:
        #    example2=copy.deepcopy(data_j['panels'][1]['targets'][0]);
        print(i,fname)
        
        var='/^$calculation$/'
        #var2='/^$calculation2$/'
        #if Type==2:
        out.append(get_query2(fname,num2alfabeta(i),var))
        #out2.append(get_query2(fname,num2alfabeta(i),var2))
        #else:    
        #    out.append(get_query(fname,num2alfabeta(i),var))
        #    out2.append(get_query(fname,num2alfabeta(i),var2))
        
    data_j['panels'][0]['targets']=out
    if len(data_j['panels'])>1:
        data_j['panels'][1]['targets']=out2

    # Проверяем присутствие переменных:
    if len(data_j['templating']['list'])==0:
        data_j['templating']=templating_list()
    
    # Сохранение модели
    with open(JsonFile[:-4]+str(Type)+"_correct.json","w", encoding='utf-8') as jsonfile:
            json.dump(data_j,jsonfile,indent=2,ensure_ascii=False)

    # Обновление схемы
    import xml.etree.ElementTree as ET
    tree = ET.parse(DrawIOFile)
    root = tree.getroot()
    data_j['panels'][0]['flowchartsData']['flowcharts'][0]['xml']=root[0].text
    #if len(data_j['panels'])>1:
    #    data_j['panels'][1]['flowchartsData']['flowcharts'][0]['xml']=root[0].text


    # Сохранение результата
    with open(JsonFile[:-4]+str(Type)+"_correct.json","w", encoding='utf-8') as jsonfile:
            json.dump(data_j,jsonfile,indent=2,ensure_ascii=False)
    jsonfile.close()
    return data_j 

def num2alfabeta(i):
    if i<26:
        out=string.ascii_uppercase[i] 
    else:
        out=string.ascii_uppercase[int(np.fix(i/26-1))]+string.ascii_uppercase[i%26] 
    return  out
