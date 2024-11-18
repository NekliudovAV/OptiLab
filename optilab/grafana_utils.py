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

