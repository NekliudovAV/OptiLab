# Файл, содержащий полезные конструкции для работы с моделями
import json
import dill
import numpy as np
import pyomo
import pickle
from pyomo.environ import *
import cloudpickle
import base64
import pandas as pd

# Список ограничений
def list_constraint(m):
    Vars=[]
    for Var1 in m.component_data_objects(Constraint):
        Vars.append(Var1.name)
    return Vars   
    
# Список целевых функций
def list_ovjective(m):
    Vars=[]
    for Var1 in m.component_data_objects(Objective):
        Vars.append(Var1.name)
    return Vars

# Список переменных
def list_vars(m):
    Vars=[]
    for Var1 in m.component_data_objects(Var):
        Vars.append(Var1.name)
    return Vars   

def get_type(aaa):
    if aaa.is_continuous():
        out='Real'
    elif aaa.is_binary():
        out='Bynary'
    elif aaa.is_integer():
        out='Integer'
    else:
        out='Unnown'
    return out    

def DF_vars(m):
    Vars=[]
    for Var1 in m.component_data_objects(Var):
        name=Var1.name
        if name.count('.')<2:
            Vars.append({'NameVar':name,'Value':Var1.value,'domain:':get_type(Var1),'IsFixed':Var1.is_fixed(),'Lb':Var1.bounds[0],'Ub':Var1.bounds[1]})
    return Vars   
    
def correct_var_name(name):
    name=name.replace('BoolVars[','')
    if 'Var' in name:
        tempName=name.split(',')
        if len(tempName)>1:
            name=tempName[1].replace(']','')
        #else:   
    #name=name.replace('Vars[0,','').replace(']','')
    name=name.replace('[','')
    name=name.replace('Turbines','')
    name=name.replace('Turbine','')
    name=name.replace('Boilers','')
    name=name.replace('REU','')
    return name        
    
def list_model_vars(m,Stages=False):
    Vars=[]
    max_point=2
    if 'Stages' in m.name:
        max_point=3
        Stages=True
    for Var1 in m.component_data_objects(Var):
        name=Var1.name
        if ('Stages'not in name) or Stages:
            if name.count('.')<max_point:
                name=correct_var_name(name)
                Vars.append(name)
    return Vars     
    
# Формирование отчёта
def gen_report(m):
        DF = pd.DataFrame()
        for v in m.component_objects(Var,active=True):
            #print(v.name)
            for index in v:
                if v.name.count('.')<3 * isinstance(index,int):
                    if v[index].value is None:
                        value=0
                        t_=index
                    elif v[index].value>=0:
                        if isinstance(index,int):
                            value=v[index].value
                            t_=index
                        elif isinstance(index,tuple):
                            value=v[index].value*np.sign(-index[1]+.5)
                            t_=index[0]
                        DF.at[t_, v.name] = value
                elif v.name.count('.')<3 * isinstance(index,tuple):
                    if v.name.find('PW')<=0:
                        var_=index[1]
                        VName=v.name
                        VName=VName.replace('.Vars', "")
                        time_=index[0]
                        DF.at[time_,VName+'.'+str(var_)] =  v[index].value
                        
                elif (v.name.count('.')<3) * (index is None):# isinstance(index,str):
                    value=v.value
                    t_=0
                    DF.at[t_, v.name] = value                     
        return DF
        
# Фильтрация имён
def gen_report2(m):
    res=gen_report(m).iloc[0:1]
    res2=res[[i  for i in res.keys() if (not 'quad' in i) and (not 'res2' in i)]]
    return res2.rename(columns={i:i.replace('SVar.','').replace('Vars.','').replace('residual.','r.') for i in res2.keys()})       
        
#Сохранение модели в файл
def save_pyomo_model(model,filename):
    strmodel=base64.b64encode(model).decode('utf-8')
    f = open(filename, "w")
    f.write(strmodel) # исправил так
    f.close()
    
# Чтение модели из файла
def load_pyomo_model(filename):    
    f = open(filename,'r')
    # работа с файлом
    res=f.readline()
    f.close()
    # Результат преобразуем в бинарную конструкцию
    res1=base64.b64decode(res)
    # Возвращаем в Pyomo
    instance = pickle.loads(res1)
    return instance

# Functions for Equation Analysis:
# Проводит анализ уравнения и если есть небаланс по исходным данным, возвращает величину небаланса
from pyomo.repn.standard_repn import generate_standard_repn
def presolve_eq(Eq):
    out=generate_standard_repn(Eq.body)
    value=out.constant
    linearVars=out.linear_vars
    QuadricVars=out.quadratic_vars
    # Линейная составляющая
    i=0
    val=0
    Koefs=out.linear_coefs
    for v in linearVars:
        if v.value is None:
            val=np.inf
            print(v.name,val,Koefs[i])
        else:
            val=v.value
        value=value+val*Koefs[i]
        i=i+1;
        
    # Квадратичная составляющая
    i=0
    Koefs=out.quadratic_coefs
    val=[0,0]
    for v in QuadricVars:
        if v[0].value is None:
            val[0]=np.inf
            print(v[0].name,v[0].value,v[1].name,v[1].value,Koefs[i])
        else:
            val[0]=v[0].value
            
        if v[1].value is None:
            val[1]=np.inf
            print(v[0].name,v[0].value,v[1].name,v[1].value,Koefs[i])
        else:
            val[1]=v[1].value
            
        
        value=value+val[0]*val[1]*Koefs[i]
        i=i+1;
    
    #print(value)
    return value

# Проверяет каждое уравнение на предмет небаланса
def analise(mdh):
    sortOrder = SortComponents.unsorted
    correct_eq=[]
    delta=[]
    for constraint_data in mdh.component_data_objects(
                            Constraint,
                            active=True,
                            sort=sortOrder,
                            descend_into=False):
        print(constraint_data.name)
        value=presolve_eq(constraint_data)
        print(constraint_data.name,' - небаланс: ',value)
        if abs(value)>0 and not np.isinf(value) :
            correct_eq.append(constraint_data)
            delta.append(value)
    return correct_eq, delta

# Функция коррекируем уравнение Eq: добавляет в него DeltaPlus и DeltaMinus Ex=Ex+DeltaPlus[j]-DeltaMinus[j]
# * mdh.DeltaPlus и mdh.DeltaMinus должны быть уже созданы в модели
def gen_Expression(mdh,Eq,j):
    out=generate_standard_repn(Eq.body)
    value=out.constant
    linearVars=out.linear_vars
    QuadricVars=out.quadratic_vars
    # Линейная составляющая
    i=0
    val=0
    Koefs=out.linear_coefs
    for v in linearVars:
        value=value+v*Koefs[i]
        i=i+1;
    # Квадратичная составляющая
    i=0
    Koefs=out.quadratic_coefs
    for v in QuadricVars:
        value=value+v[0]*v[1]*Koefs[i]
        i=i+1;
    Expr=value+mdh.DeltaPlus[j]-mdh.DeltaMinus[j]
    #print(Expr)
    mdh.Constr.add(Expr==0)     
    return Expr


#____________________ Сохранение моделей в json и их загрузка____________________


class OptimizationModel(object):
    model:    None
    operLims: None
    accuracy: None
    target =  'SE'
    fixed:    None
    strtime:  None
        
    def pprint(self):
        print('target:',self.target)
        print('strtime:',self.strtime)
        
    def calculate_SE(self):
        m2=self.model.clone()
        # Исправить на fixed
        fix_N(m2,self.modelDF,self.strtime)
        # Добавление невязок на отклонение
        m2=clsTurbineOpt.add_stat_data(m2,self.accuracy,N=32)
        set_MF2(m2,self.target)
        # Оптимальное распределение
        SEAC,status=Rubust_Calc(m2,self.strtime,self.operLims)
        return  SEAC,status       


def get_lib_versions():
    vers={}
    for lib in ['pickle','pyomo','dill','pandas','numpy']:
        vers[lib]=get_ver(lib)
    return vers

def get_ver(lib_name):
    if lib_name in ['pickle']:
        version=pickle.format_version
    elif lib_name in ['pyomo']:
        version=pyomo.__version__
    elif lib_name in ['dill']:
        version=dill.__version__
    elif lib_name in ['pandas']:
        version=pd.__version__
    elif lib_name in ['numpy']:
        version=np.__version__
    return version


def compare_lib_versions(vers_from_json):
    for lib_name in vers_from_json.keys():
        version=get_ver(lib_name)
        if vers_from_json[lib_name]==version:
            print(f'версии {lib_name} совпадают: {version}')
        else:
            print(f'сохранение выполненно в {lib_name} версии{version}, а в среде исполнения {vers_from_json[libname]}')

# Преобразованеи в Json
def df_2_json(df):
    if isinstance(df,pd.DataFrame):
        jdf=df.reset_index().to_json(orient='records')
    else:
        jdf=''
    return jdf

# Обратное преобразование
def json_2_df(jdf):
    if len(jdf)>0:
        df=pd.json_normalize(json.loads(jdf))
        if 'time' in df.keys():
            df=df.set_index('time')
            df.index=pd.to_datetime(df.index, unit='ms')
        elif 'index' in df.keys():
            df=df.set_index('index')
    else:
        df=None
    return df   
    

def accuracy_2_json(accuracy_dh):
    if isinstance(pd.DataFrame(),pd.DataFrame):
        return json.dumps(accuracy_dh)
    else:
        return ''
    
def json_2_accuracy(jAccuracy_dh):
    if len(jAccuracy_dh)>0:
        return json.loads(jAccuracy_dh)
    else:
        return None
            

def save_model(model,filename='model_custom.json',Fixed=None,ModelDF=None,OperLims=None,Accuracy=None,Target='SE',Strtime=None):
    model_json={}
    model_json['PICKLE']=base64.b64encode(dill.dumps(model)).decode('utf-8')
    model_json['ModelDF']=df_2_json(ModelDF)
    model_json['OperLims']=df_2_json(OperLims)
    model_json['Fixed']=df_2_json(Fixed)
    model_json['Accuracy']=accuracy_2_json(Accuracy)
    model_json['Target']=Target
    model_json['Strtime']=strtime.strftime('%Y-%m-%d %H:%M %Z')
    # Версии
    model_json['libs_versions']=get_lib_versions()
    with open(filename, 'w') as f:
        json.dump(model_json, f, indent=4)

def load_model_(filename='model_custom.json'):
    with open(filename, 'rb') as f:  # 'rb' — чтение в бинарном режиме
        data_json = json.load(f)
    compare_lib_versions(data_json['libs_versions'])
    pickle_data = base64.b64decode(data_json['PICKLE'].encode('utf-8'))
    # 2. Десериализуем модель из pickle
    model = pickle.loads(pickle_data)
    om=OptimizationModel()
    om.model=model
    om.modelDF= json_2_df(data_json['ModelDF'])
    om.fixed= json_2_df(data_json['Fixed'])
    om.operLims= json_2_df(data_json['OperLims'])
    om.accuracy= json_2_accuracy(data_json['Accuracy'])
    om.target = data_json['Target']
    om.strtime= pd.Timestamp(data_json['Strtime']).tz_convert(tz='Etc/GMT-3')
    return om 

# Копирование модели в Block
def _copy_component(src, dest_model):
    """Корректное копирование компонентов Pyomo между моделями/блоками"""
    # Копирование переменных
    if isinstance(src, Var) and (src.name.count('.')==0):
        
        
        if not src.is_indexed():
            new_var = Var(domain=src.domain, 
                         bounds=(src.lb, src.ub),
                         initialize=src.value)
            dest_model.add_component(src.name, new_var)
        else:
            #temp_var=Var(src.index_set())
            dest_model.add_component(src.name, Var(src.index_set()))    
            d=dest_model.__dict__
            for index in src.index_set():
                    print(str(index))
                    d[src.name][index].value = src[index].value
                    d[src.name][index].lb = src[index].lb
                    d[src.name][index].ub = src[index].ub
                    d[src.name][index].domain = src[index].domain
               
        
    # Копирование блоков
    elif isinstance(src, Block):
        # блоки копируются только целиком
        if not src.is_indexed():
            new_block = Block(concrete=True)
            if src.name.count('.')==0:
                print(f'Block:',src.name)
                dest_model.add_component(src.name, src.clone())


    # Копирвание ограничений
    elif isinstance(src, Constraint):
        print(f'Constraint:',src.name)
        if not src.is_indexed():
            print('Constraint not indexes')
            new_constr = Constraint(expr=src.expr)
            dest_model.add_component(src.name, new_constr)
        else:
            print('Constraint indexed')
            new_constr = Constraint(src.index_set())
            dest_model.add_component(src.name, new_constr)
            for index in component:
                 new_constr[index] = src[index].expr
        
    elif isinstance(src, Objective):
        print(f'Objective:',src.name)
        new_obj = Objective(expr=src.expr, sense=src.sense)
        dest_model.add_component(src.name, new_obj)
        return new_obj

''' # Заготовка объединения моделей в единую
        merged_model = ConcreteModel()
        merged_id = str(uuid.uuid4())
        model_id='KemGres'
        # 2. Обрабатываем каждую исходную модель
        for src_model in [model]:

            #src_model = models_db[model_id]["pyomo_model"]
            
            # 3. Создаем блок для подмодели
            block_name = f"{model_id}"
            block = Block(concrete=True)
            merged_model.add_component(block_name, block)
            
            # 4. Копируем все компоненты
            for component in src_model.component_objects():
                if component.name.count('.')==0:
                    print(f'Копируется компонент {component.name}:')
                    _copy_component(component, block)

            # 5. Добавляем связующие ограничения
            for constr in request.linking_constraints:
                expr = evaluate_expression(constr["expression"], {}, merged_model.__dict__)
                merged_model.add_component(
                    constr["name"], 
                    Constraint(expr=expr)
                )
            
            # 6. Сохраняем объединенную модель
            models_db[merged_id] = {
                "pyomo_model": merged_model,
                "components": request.model_ids,
                "variable_map": request.variable_mapping
            }'''
