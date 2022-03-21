from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import cloudpickle
import base64

def create_mode(dataB,nhn):
    # nhn - number hiddeb neuron
    sc = StandardScaler(dataB)
    sc_dataB = sc.fit_transform(dataB)
    nn = MLPRegressor(nhn, verbose=True,
                  learning_rate_init=0.0001,
                  learning_rate='adaptive', 
                  alpha=0.1, tol=1e-12, 
                  max_iter=30000, max_fun=1e-8,
                  n_iter_no_change= 100)
    return nn,sc

def fit(nn,sc,dataB)
    sc_dataB = sc.fit_transform(dataB)
    nn.fit(sc_dataB[:,:-1], sc_dataB[:,-1])
    new_dataB = pd.DataFrame(sc.inverse_transform(sc_dataB), columns=dataB.columns)
    # расчёт погрешности
    m_err=(dataB.y-new_dataB.y).abs().mean()
    m_max=(dataB.y-new_dataB.y).abs().mean()
    return new_dataB, m_err, m_max

# Save to file in the current working directory
def save_nn(nn,path2file):
    joblib.dump(nn, path2file)
     
def load_nn(path2file):
    nn = joblib.load(path2file)
    return nn

# Сохранение набора данных: нейросеть плюс маштабирование
def save_nn_sc(nn,sc,path2file):
    data={nn:nn,sc:sc};
    strmodel=base64.b64encode(nn).decode('utf-8')
    f = open(filename, "w")
    f.write(strmodel) # исправил так
    f.close()
    
# Загрузка набора данных: нейросеть плюс маштабирование
def load_nn_sc(path2file)
    f = open(filename,'r')
    # работа с файлом
    res=f.readline()
    f.close()
    # Результат преобразуем в бинарную конструкцию
    res1=base64.b64decode(res)
    # Возвращаем в Pyomo
    instance = pickle.loads(res1)
    return instance['nn'],instance['sc']