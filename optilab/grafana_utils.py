import json
import os
import codecs
import pandas as pd
import numpy as np
import string
import copy
import re 
import xml.etree.ElementTree as ET
import base64
import zlib
from urllib.parse import unquote

#Достать 
def undecode_compressed(BinaryJson):
    decoded_data=base64.b64decode(BinaryJson)
    decompressed_data = zlib.decompress(decoded_data, wbits=-15)
    final_xml_string = unquote(decompressed_data.decode('utf-8'))
    return final_xml_string

def get_B_drawio_xml_string(JsonFile):
    with codecs.open(JsonFile, "r","utf-8") as json_file:
        data_j=json.load(json_file)
    BinaryJson=data_j['panels'][0]['flowchartsData']['flowcharts'][0]['xml']
    return BinaryJson
# DataFile=".\Grafana\T_11\TA11.xlsx"
# BJson=get_B_drawio_xml_string(JsonFile)
# correct_Gr_Json(JsonFile,DataFile,BJson)    

 

def compress_xml_for_drawio(xml_string):
    """
    Compresses an XML string for use in a .drawio file.
    """
    # 1. Compress the XML string using zlib deflate (raw stream)
    obj=zlib.compressobj(level=9,wbits=-15)
    #compressed_data=obj.compress(xml_string.encode('utf-8'))+obj.flush()
    compressed_data=obj.compress(xml_string.encode('utf-8'))+obj.flush()

    # 2. Base64 encode the compressed data
    base64_encoded_data = base64.b64encode(compressed_data)#.decode('utf-8')

    # 3. Create the .drawio XML structure
    # This is a simplified example; a real draw.io file has more attributes
    root = ET.Element("mxfile", host="Python", agent="Python-Script")
    diagram = ET.SubElement(root, "diagram", id="myDiagram", name="Page-1")
    diagram.text = base64_encoded_data

    # Convert the ElementTree to a string
    return base64_encoded_data #ET.tostring(root, encoding='utf-8').decode('utf-8')
#compress_xml_for_drawio(DrawIO).decode('utf-8')
    
#JsonFile=".\Grafana\Boilernaya\Grafana_Json.json"
def get_drawio_xml_string(JsonFile):
    with codecs.open(JsonFile, "r","utf_8_sig") as json_file:
        data_j=json.load(json_file)
    BinaryJson=data_j['panels'][0]['flowchartsData']['flowcharts'][0]['xml']
    return undecode_compressed(BinaryJson)
# root = ET.fromstring(final_xml_string)


# Сохранение таблицы с переменными DrawIO
def Draio2Table(final_xml_string):
    if len(final_xml_string)<100:
        tree = ET.parse(DrawIOFile)
        root = tree.getroot()
    else:    
        root = ET.fromstring(final_xml_string)
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
            Table.append(pd.DataFrame({'id':[id],'Переменная':[Value],'Добалвение текста':['anl'],'Цвет':[' '],'Тип фигуры':[figure]}))
    Table=pd.concat(Table)
    Table=Table.reset_index().drop(columns=['index'])
    #Table.to_excel(XlsFile)
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
   
def get_query2(fname='КА11.D0L2',refId='A',var='typecalc'):
    if var in ['typecalc']:
        # Тип запроса 1
        #query=f'SELECT mean(\"{fname}\") FROM /^$calculation$/ WHERE  (\"Station\"::tag =~/^$station$/) AND (\"Equipment\"::tag =~/^$equipment$/) AND (\"TypeCalc\"::tag =~/^$'+var+'$/) AND $timeFilter GROUP BY time($__interval) fill(none)'
        query=f'SELECT mean(\"{fname}\") FROM /^$calculation$/ WHERE ("Ni"::tag =~ /^$Ni$/ AND "fleet"::tag =~ /^$fleet$/) AND ("n_boilers"::tag =~ /^$nBoilers$/) AND $timeFilter GROUP BY time($__interval) fill(none)'
    else:    
        # Тип запроса 2
        
        query=f'SELECT mean(\"{fname}\") FROM /^$calculation$/ WHERE  (\"Station\"::tag =~/^$station$/) AND (\"Equipment\"::tag =~/^$equipment$/) AND (\"TypeCalc\"::tag =~/^$'+var+'$/) AND ("Model"::tag =~/^$model$/) AND ("Scenario"::tag =~/^$scenario$/) AND ("Version"::tag =~/^$version$/) AND $timeFilter GROUP BY time($__interval) fill(none)'
    
    temp={'alias': fname.replace('.','_'),
     'groupBy': [{'params': ['$__interval'], 'type': 'time'},
      {'params': ['null'], 'type': 'fill'}],
     'orderByTime': 'ASC',
     'policy': 'default',
     'query': query,
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
    
def get_var(query="SHOW MEASUREMENTS",label="Данные",name="calculation",uid="ff760b74-f5c8-4935-a467-655d48f3e022"):
    null=None
    false=False
    true=True
    u_list={
        "current": {
          "selected": false,
          "text": "Analise",
          "value": "Analise"
        },
        "datasource": {
          "type": "influxdb",
          "uid": uid
        },
        "definition": query,
        "hide": 0,
        "includeAll": false,
        "label": label,
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
    return u_list
    
    
def templating_list2(path2file='.\Grafana\Boilernaya\Boilernaja.xlsx',sheet_name='Vars'):
    dataVars=pd.read_excel(path2file,sheet_name=sheet_name)
    l=[]
    for i in range(dataVars.shape[0]):
        temp=dataVars.iloc[i]
        l.append(get_var(query=temp['query'],label=temp['label'],name=temp['name']))
    return {"list":l}
#templating_list2(path2file=DataFile)
    

def correct_Gr_Json(JsonFile,DataFile,DrawIO,Type=2):
    # Чтение данных модели
    with codecs.open(JsonFile, "r","utf_8_sig") as json_file:
        data_j=json.load(json_file)

    # Чтение шаблона
    import copy
    # Берём тэги из Файла:
    data=pd.read_excel(DataFile,sheet_name='Правила')
    data.head()

    # Формирование правил данных и добаление из в json
    out=[]
    out2=[]
    for i in data.index:
        example1=copy.deepcopy(data_j['panels'][0]['rulesData']['rulesData'][0]);
        example2=copy.deepcopy(data_j['panels'][0]['rulesData']['rulesData'][0]);
        fname=data['Переменная'][i]
        shape_color=data['Цвет'][i]
        
        # 
        for id_cell_column in ['Показывать значение','id']:
            if  id_cell_column in data.keys():
                text=data[id_cell_column][i]
            
        add_text=data['Добалвение текста'][i]
        print(i,fname,shape_color,text)
        
        out.append(get_rools(fname,shape_color,text,add_text))
        out2.append(get_rools(fname,shape_color,text,add_text))
        
    # Перезапись в Json
    data_j['panels'][0]['rulesData']['rulesData']=out
    if len(data_j['panels'])>1:
        data_j['panels'][1]['rulesData']['rulesData']=out2

    #import copy

    data_t=data[['Переменная']]
    data_t=data_t.rename(columns = {'Переменная':'value'})


    out=[]
    out2=[]
    for i, fname in enumerate(data_t.value):
        example1=get_example()
        if len(data_j['panels'])>1:
            example2=copy.deepcopy(data_j['panels'][1]['targets'][0]);
        print(i,fname)
        

        out.append(get_query2(fname,num2alfabeta(i),var='typecalc'))
        out2.append(get_query2(fname,num2alfabeta(i),var='typecalc_r'))

    data_j['panels'][0]['targets']=out
    if len(data_j['panels'])>1:
        data_j['panels'][1]['targets']=out2

    # Проверяем присутствие переменных:
    #if len(data_j['templating']['list'])==0:
    data_j['templating']=templating_list2(path2file=DataFile,sheet_name='Vars')
        
    
    # Сохранение модели
    with open(JsonFile[:-4]+str(Type)+"_correct.json","w", encoding='utf-8') as jsonfile:
            json.dump(data_j,jsonfile,indent=2,ensure_ascii=False)

    
    
    if len(DrawIO)<100:
        tree = ET.parse(DrawIO)
        root = tree.getroot()
        textDrawIO=root[0].text
    else:    
        textDrawIO = DrawIO
    # Обновление схемы
    data_j['panels'][0]['flowchartsData']['flowcharts'][0]['xml']=textDrawIO
    if len(data_j['panels'])>1:
        data_j['panels'][1]['flowchartsData']['flowcharts'][0]['xml']=textDrawIO


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
# Пример вызова    
# JsonFile='TA8.json'
# DataFile='TA8_aouto.xlsx'
# DrawIOFile='TA8.xml'
#correct_Gr_Json(JsonFile,DataFile,DrawIOFile,Type=1)

