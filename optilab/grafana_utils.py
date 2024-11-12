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

# работа с деревом xml: https://docs.python.org/3/library/xml.etree.elementtree.html
# формируем список элементов:
temp=[]
for child in root:
    print('tag: ',child.tag,'atrib: ', child.attrib)
    if 'value' in child.attrib.keys():
        temp.append({'tag':child.tag, 'id':child.attrib['id'],'value':child.attrib['value']})
    else:
        temp.append({'tag':child.tag, 'id':child.attrib['id'],'value':''})

def correct_Gr_Json(JsonFile,DataFile,DrawIOFile,Type=1):
    # Чтение данных модели
    with codecs.open(JsonFile, "r","utf_8_sig") as json_file:
        data_j=json.load(json_file)

    # Чтение шаблона

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
        text=data['Показывать значение'][i]
        add_text=data['Добалвение текста'][i]
        print(i,fname,shape_color,text)
        
        out.append(get_rools(fname,shape_color,text,add_text))
        out2.append(get_rools(fname,shape_color,text,add_text))
        
        #out.append(gen_rulesData(example1,fname,shape_color,text,add_text))
        #out2.append(gen_rulesData(example2,fname,shape_color,text,add_text))
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
        #example1=copy.deepcopy(data_j['panels'][0]['targets'][0]);
        example1=get_example()
        if len(data_j['panels'])>1:
            example2=copy.deepcopy(data_j['panels'][1]['targets'][0]);
        print(i,fname)
        
        var='/^$calculation$/'
        var2='/^$calculation2$/'
        if Type==2:
            out.append(get_query2(fname,num2alfabeta(i),var))
            out2.append(get_query2(fname,num2alfabeta(i),var2))
            #out.append(gen_targets(example1,fname,num2alfabeta(i),var))
            #out2.append(gen_targets(example2,fname,num2alfabeta(i),var2))
            #out.append(gen_targets(example,fname,num2alfabeta(i)))
        else:    
            out.append(get_query(fname,num2alfabeta(i),var))
            out2.append(get_query(fname,num2alfabeta(i),var2))
        
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
    if len(data_j['panels'])>1:
        data_j['panels'][1]['flowchartsData']['flowcharts'][0]['xml']=root[0].text


    # Сохранение результата
    with open(JsonFile[:-4]+str(Type)+"_correct.json","w", encoding='utf-8') as jsonfile:
            json.dump(data_j,jsonfile,indent=2,ensure_ascii=False)
    jsonfile.close()
    return data_j 
