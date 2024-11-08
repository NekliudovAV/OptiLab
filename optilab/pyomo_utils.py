# Файл, содержащий полезные конструкции для работы с моделями
import pyomo
import pickle
from pyomo.environ import *
import cloudpickle
import base64

# Список ограничений
def list_constraint(m):
    Vars=[]
    for Var1 in m.component_data_objects(Constraint):
        Vars.append(Var1.name)
    return Vars   
    
# Список целефых функций
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
