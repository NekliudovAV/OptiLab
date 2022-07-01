import pickle
from pyomo.environ import *
import cloudpickle
import base64

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