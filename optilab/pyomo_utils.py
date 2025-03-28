# Файл, содержащий полезные конструкции для работы с моделями
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
