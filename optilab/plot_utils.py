# Отображение в диаграмме рассеивания
# Вспомогательная функция
import warnings
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
import inspect

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

def get_simplexes(df_):
    fig_ = go.Figure()
    if isinstance(unit_specification['block_equations']['eq_Dpvd'],dict):
        df_=pd.DataFrame(data=df_)
    k=1.8
    VarNames=list(df_.keys())
    VarNames=VarNames[0:-1]

    title_=VarNames[-1]

    num_simp=0
    for t in df_['simplex_numb'].values:
        num_simp=max(num_simp,(max(t)))
    sp=[[] for i in range(num_simp+1)]
    i=0
    camera = dict(eye=dict(x=k, y=0.3, z=k)) # формирование словаря
    for t in df_['simplex_numb'].values:
        for t_ in t:
            sp[t_].append(i)
        #num_simp=max(num_simp,(max(t)))
        i=i+1
    if len(VarNames)<3:    
        type_plot= 'scatter'
    else:
        
        type_plot= 'surface'
    fig = make_subplots(
        rows=1, cols=2,specs=[[{'type': type_plot},{'type':'table'}]],figure=fig_)      

    for ind in sp:
        #index=dindex # выбор индекса, соответствующего уникальному значению и выполнение среза
        dfI=df_
        index=ind
        # создаем 3D объект на основе характеристик данных, полученных функцией и размещаем, созданный объект в первом столбце
        if len(VarNames)<3:
            fig.add_trace(go.Scatter(x=dfI[VarNames[-2]].iloc[index].values,y=dfI[VarNames[-1]].iloc[index].values,
                                      opacity=0.9,hovertext=VarNames[-1]),row=1,col=1)

        else:
            fig.add_trace(go.Mesh3d(x=dfI[VarNames[-3]].iloc[index],y=dfI[VarNames[-2]].iloc[index],z=dfI[VarNames[-1]].iloc[index],
                                colorscale="BuGn", opacity=0.9,hovertext=VarNames[-1]),row=1, col=1)
            #fig.update_xaxes(range=[dfI[VarNames[-2]].min(),dfI[VarNames[-2]].max])
            #fig.update_yaxes(range=[dfI[VarNames[-1]].min(),dfI[VarNames[-1]].max]) 
    if len(VarNames)<3:
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
    dfI_=dfI.round(2)
    fig.add_trace(go.Table(
        header=dict(values=list(dfI_.columns), align='left'),
        cells=dict(values=dfI_.values.transpose(), align='left')),
                  row=1, col=2)
    #fig.show()
    return fig#None
