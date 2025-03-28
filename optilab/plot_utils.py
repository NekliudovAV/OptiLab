# Отображение в диаграмме рассеивания
# Вспомогательная функция
import warnings
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
import inspect
from optmodel_utils import get_sigma_U
import json

# Необходимо подготовить функицию, демонстрирующую характеристики оборудования:
import plotly.graph_objects as go # импортируем библиотеку graph_objects для 3D отображения
from plotly.subplots import make_subplots # подключаем возможность разбиения блока на два блока

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

def plor_diag_ras(DF_Fact,DF_Estiate,accuracy_dh=None):
    # Диаграмма рассеивания. 
    
    # Добавлены проверки на исходные данные
    if accuracy_dh==None:
        accuracy_dh=get_sigma_U(DF_Estiate)
        
    # Определяем столбцы, которые присутствую в данных и в результатах
    res_keys=DF_Estiate.keys()
    df_keys=DF_Fact.keys()
    columns=list(set(res_keys).intersection(df_keys))
    DF_Fact=DF_Fact[columns]
    DF_Estiate=DF_Estiate[columns]
    
    # Выбираем только те интервалы времени, по которым расчёты успешно завершились
    DF_Fact=DF_Fact.loc[DF_Estiate.index]
    
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


def plot_df(title_,dfI):
# Отображение датафреймов    
    dfI=dfI.copy()
    dfI_table=dfI.copy()
    # Корректируем размерность df для отображения:
    # Если параметр в колонке не менятеся, удаляем колонку
    for k in dfI.keys():
        if len(dfI[k].unique())==1:
            dfI=dfI.drop(columns=[k])
    #
    k=1.8
    VarNames=list(dfI.keys()) # формирование списка
    camera = dict(eye=dict(x=k, y=0.3, z=k)) # формирование словаря
    # разбиваем графическую область на одну строку и дву колонки
    if len(VarNames)<3:    
        type_plot= 'scatter'
    else:
        type_plot= 'surface'
        
    fig = make_subplots(
        rows=1, cols=2,specs=[[{'type': type_plot},{'type':'table'}]])

    # Отрисовка характеристики
    # Выбор уникального значения по определенному параметру
    if dfI.shape[1]>3:
        for tDm in dfI[VarNames[-4]].unique():
            index=dfI[VarNames[-4]]==tDm # выбор индекса, соответствующего уникальному значению и выполнение среза
            # создаем 3D объект на основе характеристик данных, полученных функцией и размещаем, созданный объект в первом столбце
            fig.add_trace(go.Mesh3d(x=dfI[VarNames[-3]][index],y=dfI[VarNames[-2]][index],z=dfI[VarNames[-1]][index],
                                    colorscale="BuGn", opacity=0.9,hovertext=VarNames[0]+'='+str(tDm)),row=1, col=1)
    elif dfI.shape[1]==3:
            fig.add_trace(go.Mesh3d(x=dfI[VarNames[-3]],y=dfI[VarNames[-2]],z=dfI[VarNames[-1]],
                                    colorscale="BuGn", opacity=0.9,hovertext=VarNames[0]+'='+str('')),row=1, col=1)
    elif dfI.shape[1]<3:
            fig.add_trace(go.Scatter(x=dfI[VarNames[-2]].values,y=dfI[VarNames[-1]].values,
                                      opacity=0.9,hovertext=VarNames[-1]),row=1,col=1)
        
            
    if dfI.shape[1]<3:
        fig.update_layout(scene = dict(
                        xaxis_title=VarNames[-2],
                        yaxis_title=VarNames[-1]),showlegend = True,
                     scene_camera=camera,width=1000,height=600,title=title_)    
    else:    
        fig.update_layout(scene = dict(
                        xaxis_title=VarNames[-3],
                        yaxis_title=VarNames[-2],
                        zaxis_title=VarNames[-1]),showlegend = True,
                     scene_camera=camera,width=1000,height=600,title=title_)    
        
    # Отрисовка таблицы
    fig.add_trace(go.Table(
        header=dict(values=list(dfI_table.columns), align='left'),
        cells=dict(values=dfI_table.values.transpose(), align='left')),
                  row=1, col=2)
    #fig.show()
    return fig

def sort_columns_by_uvalues(Qt):
    df={}
    for k in Qt.keys():
        df[k]=[len(Qt[k].unique())]
    df=pd.DataFrame(df).transpose()
    return Qt[list(df.sort_values(by=[0]).index)]

def slice_(Qt,fix_values=None):
    if fix_values==None:
        Fixed_fils=['H0','T0','P0', 'Dd','P2','IWF','dGfwD0','Dsp','IDsp','Pmix']
        fix_values=set(Fixed_fils).intersection(Qt.keys())

    F=[Qt.keys()[-1]]
    lm = LinearRegression()
    lm.fit(Qt.iloc[:,:-1],Qt.iloc[:,-1])
    # Расчёт погрешности:
    Fit=lm.predict(Qt.iloc[:,:-1])
    print('Погрешность slice-ера, %',((Qt.iloc[:,-1]-Fit)/Qt.iloc[:,-1]).abs().mean())
    
    Data=Qt.copy()
    for k in fix_values:
        Data[k]=Data[k][Data[k]!=0].mean()
    t=(set(Data.keys())-set(fix_values))-set(F) 
    t=list(t)
    t.extend(F)

    Keys=list(Data.keys())
    Fit=lm.predict(Data[Keys[:-1]])
    Data[Keys[-1]]=Fit
    Data=Data.sort_values(by=F)
    return Data

def plot_df_dim(df,fix_values=None,Name=''):
    if df.shape[1]>4:
        df=slice_(df,fix_values=None)
    df=sort_columns_by_uvalues(df)
    return plot_df(f'Харакетристика:{Name}',df)
