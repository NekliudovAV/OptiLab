import json
import os
import codecs
import pandas as pd
import numpy as np
import string
import copy

# Открытие файла Json
JsonFile='teploset.json'
with codecs.open(JsonFile, "r","utf_8_sig") as json_file:
        data_j=json.load(json_file)

# Правила отображения данных
rulesData=copy.deepcopy(data_j['panels'][0]['rulesData']['rulesData'][0]);
temp=[]
for trule in data_j['panels'][0]['rulesData']['rulesData']:
    shapeData=trule['shapeData']
    if len(shapeData)==0:
        shapeData=""
    elif "pattern" in shapeData[0].keys():
        shapeData=shapeData[0]["pattern"]
    temp.append(pd.DataFrame({'Rule name':[trule['alias']],'Apply to metrix':[trule['pattern']],'Показывать значение':[trule['textData'][0]['pattern']],'Добавление текста':[trule['textData'][0]['textReplace']],'shapeData':[shapeData]}))
res=pd.concat(temp)    
res[['Rule name','Показывать значение','Добавление текста','Apply to metrix','shapeData']].to_excel('rules.xlsx')

# сохранение правил в формате xlsx
targets=[]
for target in data_j['panels'][0]['targets']:
    if 'query' in target.keys():
        targets.append(pd.DataFrame({'alias':[target['alias']],'query':[target['query']],'refId':[target['refId']]}))
    else:
        targets.append(pd.DataFrame({'alias':[target['alias']],'query':[target['select'][0][0]['params']],'refId':[target['refId']]}))
targets=pd.concat(targets)    
targets.to_excel('targets.xlsx')
