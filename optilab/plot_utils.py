# Отображение в диаграмме рассеивания
# Вспомогательная функция
import warnings
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
import inspect

def plot_dr(ax,DF_Fact,DF_Estiate,Name='D0',ci=0.2,color='green'):
    x=DF_Estiate[Name]
    y=DF_Fact[Name]
    # Генерация случайных данных
    
    x0= np.array([min(x),max(x)])
    y0= x0
    
    # Рассчитываем доверительные интервалы
    y_upper = y0 + ci
    y_lower = y0 - ci
    
    # Строим диаграмму рассеивания
    ax.scatter(x, y, label=Name)
    ax.plot(x0,y0, label='Идеальная линия', color='red')
    
    # Строим доверительные интервалы
    ax.fill_between(x0, y_lower, y_upper,  alpha=0.1, color=color, label='Доверительный интервал 2 сигма')
    
    # Добавляем подписи и легенду
    ax.set_xlabel('Скорректированные занчения')
    ax.set_ylabel('Исходные значения')
    ax.set_title('Диаграмма рассеивания '+Name)
    ax.legend()

def plor_diag_ras(DF_Fact,DF_Estiate,accuracy_dh):
    n=DF_Fact.shape[1]
    ny=int(np.fix((n+1)/2))
    fig, axs = plt.subplots(ny, 2, figsize=(10, ny*5))
    # Установим случайное семя для воспроизводимости
    Names=DF_Fact.keys()
    for i in range(n):
        #print(np.fix((i+1)/2),(i-1)%2,Names[i])
        Name=Names[i]
        plot_dr(axs[int(np.fix((i)/2)),(i)%2],DF_Fact,DF_Estiate,Name=Name,ci=accuracy_dh[Name]['s'])
    # Отображаем график
    #plt.tight_layout()
    plt.show()
