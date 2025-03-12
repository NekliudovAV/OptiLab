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
    