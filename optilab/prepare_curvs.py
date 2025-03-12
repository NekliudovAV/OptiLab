import pandas as pd
import numpy as np
from seuif97 import *
import datetime
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from scipy.spatial  import ConvexHull
from pyomo.environ import *
from df2block import *
from database import *

import plotly.graph_objects as go # импортируем библиотеку graph_objects для 3D отображения
from plotly.subplots import make_subplots # подключаем возможность разбиения блока на два блока
from sklearn.linear_model import LinearRegression




def calc_H(P,T):
        Hs=pd.Series([pt2h((P[i]+1)* 0.0980665,T[i])  for i in T.index],T.index)
        Hs=Hs/4.186
        Hs_=pd.Series([tx2h(T[i],1) for i in T.index],T.index)
        Hs_=Hs_/4.186
        Hs[Hs_>Hs]=Hs_[Hs_>Hs]
        return Hs
    
def save_curve4(Data,ch=[]):
        # Первая колонка является регрессором
        Data=Data.copy()
        Keys=Data.keys()
        Data=Data[list(Keys[1:])+[Keys[0]]]
        
        
        Data.reset_index(drop= True , inplace= True )
        LastRow=Data.iloc[:,-1]
        Keys=Data.keys()
        lm = LinearRegression()
        
        lm.fit(Data.iloc[:,:-1],Data.iloc[:,-1])
        lK=len(Keys)
        
        if len(ch)==0:
            chKeys=list(set(Keys[0:-1]).intersection(set(['N','Qt'])))
        else:
            chKeys=list(set(Keys[0:-1]).intersection(ch))
            
        print('chKeys:',chKeys)
        if len(chKeys)>1:
            b = ConvexHull(Data[chKeys])
            vertices=b.vertices
        else:
            vertices=np.array([Data[chKeys].idxmin(),Data[chKeys].idxmax()])
            
        # Возвращаем аппраксимирующую поверхность
        Fit=lm.predict(Data[Keys[:-1]])
        
        delta=Fit-Data[Keys[-1]]
        print(LastRow.min())
        ind=LastRow>0
        print('MAEr: ',np.mean(abs(delta[ind])),', std:',np.std(delta[ind]), ', MAPE:', (delta[ind]/LastRow[ind]).abs().mean()*100,'%')
        Data[Keys[-1]]=Fit

        # Выбираем минимальные и максимальные значения
        if len(chKeys)>1:
            irows=np.unique(np.concatenate((Data.idxmax().values,Data.idxmin().values, vertices)))
        elif len(chKeys)<=1:
            irows=np.unique(np.concatenate((Data.idxmax().values,Data.idxmin().values)))

        return Data.iloc[list(irows)]

def Data_Fit(Tdata):
        warnings.filterwarnings('ignore')
        ks=Tdata.keys()
        Values = Tdata.values
        ValsX = Values[:,1:]
        ValsF = Values[:,0]

        lm = LinearRegression()
        lm.fit(ValsX,ValsF)
        print('коеффициенты',lm.coef_)
        print('intercept',lm.intercept_)
        Tdata['fit']=lm.predict(ValsX)
        Tdata['Error+2']=Tdata[ks[0]]-Tdata['fit']
        sigma=Tdata['Error+2'].std()
        Tdata['2sigma+']=2*sigma
        Tdata['2sigma-']=-2*sigma
        print('сигма :',sigma,' MAPE (%):', (Tdata['Error+2']/Tdata[ks[0]]).abs().mean()*100)


        #Tdata[['fit',ks[0],'Error+2','2sigma+','2sigma-']].plot(figsize=(42,5))
        # Перерисовать:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=Tdata.fit.index, y=Tdata.fit,
                        mode='lines',
                        name='Fit'),row=1, col=1)

        fig.add_trace(go.Scatter(x=Tdata[ks[0]].index, y=Tdata[ks[0]],
                        mode='lines',
                        name='Base'),row=1, col=1)

        fig.add_trace(go.Scatter(x=Tdata['Error+2'].index, y=Tdata['Error+2'],
                        mode='lines',
                        name='Error'),row=1, col=1)

        fig.add_trace(go.Scatter(x=Tdata['2sigma+'].index, y=Tdata['2sigma+'],
                        mode='lines',
                        name='2sigma+'),row=1, col=1)

        fig.add_trace(go.Scatter(x=Tdata['2sigma-'].index, y=Tdata['2sigma-'],
                        mode='lines',
                        name='2sigma-'),row=1, col=1)
        fig.show()
        Tdata2=Tdata[np.abs(Tdata[ks[0]]-Tdata['fit'])<2*sigma]
        filt_=np.abs(Tdata[ks[0]]-Tdata['fit'])<2*sigma
        #warnings.filterwarnings("default")
        return Tdata['fit'],Tdata2,filt_


    def extend_ch_df(ch_df,add_df,ch_name):

        lm = LinearRegression()
        lm.fit(ch_df.iloc[:,:-1],ch_df.iloc[:,-1])

        ext_df = ch_df.copy()
        Data = pd.concat([ext_df,add_df.reset_index()]).reset_index()[ch_df.keys().intersection(add_df.keys())]


        if len(ch_name)>1:
            b = ConvexHull(Data[ch_name])
            irows=np.unique(np.concatenate((Data.idxmax(),Data.idxmin(), b.vertices)))
        else:
            irows=np.unique(np.concatenate((Data.idxmax(),Data.idxmin())))
            
        # Возвращаем аппраксимирующую поверхность
        Keys=ch_df.keys()
        Fit=lm.predict(Data[Keys[:-1]])
        Data[Keys[-1]]=Fit
        return Data.iloc[list(irows)].reset_index(drop=True)

def plot_corrected(out,MData,cols):
        fig = make_subplots(rows=round(((len(cols)-1)-0.1)/2)+1, cols=2)


        for i,name in enumerate(cols):
            #print(round((i-0.1)/2)+1,(i%2)+1)
            fig.add_trace(
                go.Scatter(x=[MData[name].min(),MData[name].max()], y=[MData[name].min(),MData[name].max()],line=dict(color="#ff0000"),name='Идеально'),
                row=round((i-0.1)/2)+1, col=(i%2)+1  )


            fig.add_trace(
                go.Scatter(x=out[name], y=MData[name],opacity=0.8,mode = 'markers',hovertext=list(MData.index.strftime('%m-%d-%Y %H:%M')) ,name=name),
                row=round((i-0.1)/2)+1, col=(i%2)+1  )

        n=len(cols)
        fig.layout.xaxis1.title={'text':'Скорректировваное'}
        fig.layout.yaxis1.title={'text':'Измеренные'}
        if n>1:
            fig.layout.xaxis2.title={'text':'Скорректировваное'}
            fig.layout.yaxis2.title={'text':'Измеренные'}
        if n>2:    
            fig.layout.xaxis3.title={'text':'Скорректировваное'}
            fig.layout.yaxis3.title={'text':'Измеренные'}
        if n>3:    
            fig.layout.xaxis4.title={'text':'Скорректировваное'}
            fig.layout.yaxis4.title={'text':'Измеренные'}
        if n>4:    
            fig.layout.xaxis5.title={'text':'Скорректировваное'}
            fig.layout.yaxis5.title={'text':'Измеренные'}
        if n>5:    
            fig.layout.xaxis6.title={'text':'Скорректировваное'}
            fig.layout.yaxis6.title={'text':'Измеренные'}




        fig.update_layout(height=600, width=800, title_text="Измеренные и скорректированные")
        fig.show()





def sort_columns_by_uvalues(Qt):
    df={}
    for k in Qt.keys():
        df[k]=[len(Qt[k].unique())]
    df=pd.DataFrame(df).transpose()
    return Qt[list(df.sort_values(by=[0]).index)] 
    
def plot_df(title_,dfI):
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
            fig.add_trace(go.Mesh3d(x=dfI[VarNames[-3]],y=dfI[VarNames[-2]],z=dfI[VarNames[-1]],colorscale='amp',
                                     opacity=0.9,hovertext=VarNames[0]+'='+str('')),row=1, col=1)
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
    
def plot_df_dim(df,fix_values=None,Name='',sort_Columns=True):
    # вЫПОЛНЯЕТ ОТРИСОВКУ ХАРАКЕТРИСТИК
    print('shape df:',df.shape[1])
    if df.shape[1]>4:
        df=slice_(df,fix_values=None)
    if sort_Columns:    
        df=sort_columns_by_uvalues(df)
    df=df.round(1)    
    return plot_df(f'Характeристика:{Name}',df)    
