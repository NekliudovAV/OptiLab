from pyomo.environ import *
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core import (Block,
                        Var,
                        Constraint,
                        RangeSet,
                        NonNegativeReals,
                        Binary)
import pandas as pd
import numpy as np
import numbers
from sklearn.linear_model import LinearRegression
from scipy.spatial import Delaunay, ConvexHull
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler  
from scipy.interpolate import interp1d, LinearNDInterpolator
    

    
def add_dN(b,VarName):
    t=b.t
    dN=30
    N_min=50
    VarName='N'
    
    
    if VarName in b.VarNames: 
        
        dVarname=[VarName+'up',VarName+'down'];
        b.dVar=Var(t,dVarname,domain=NonNegativeReals)
        
        for t_ in t[:-1]:
            b.q_delta.add(b.dVar[t_,dVarname[0]] <= N_min * (1 - b.State) + b.State * dN)
            b.q_delta.add(b.dVar[t_,dVarname[1]] <= N_min * (1 - b.State) + b.State * dN)
            b.q_delta.add(b.dVar[t_,dVarname[0]]-b.dVar[t_,dVarname[1]]==b.Vars[t_+1,VarName]-b.Vars[t_,VarName])
            

def NN_Block(t, dfI, nn, scaler, **varargs):
    # nn     # Структура нейронной сети
    # additionalvar    # Ограничения на переменные

    dfs = dfI.copy()
    nn_c0 = np.repeat(nn.coefs_[0][np.newaxis, :, : ], len(t), axis=0)
    nn_i0 = np.repeat(nn.intercepts_[0][np.newaxis, : ], len(t), axis=0)
    nn_c1 = nn.coefs_[1]
    nn_i1 = nn.intercepts_[1]
    sc_mn = scaler.mean_
    sc_sc = scaler.scale_
    
    if  'pinned' in varargs:
        extD=varargs['pinned'] 
        try:
            ext_df = pd.DataFrame({ key: pd.Series(value, index=t) for key, value in extD.items() })
            ext_df = pd.concat([ext_df[[col]] for col in dfs.columns if col in ext_df.columns], axis=1)
        except:
            raise Exception('Length of pinned values should be one or equals to the number of problem time periods')
        
        ind = [dfs.columns.get_loc(c) for c in dfs.columns if c in extD.keys()]
        
        for lt in t:
            
            fixed_vals = (ext_df.loc[lt] - sc_mn[ind]) / sc_sc[ind]

            nn_i0[lt] += fixed_vals @ nn_c0[lt,ind,:]
        
        dfs.drop(extD.keys(), axis=1, inplace=True)    
        sc_mn = np.delete(sc_mn, ind)
        sc_sc = np.delete(sc_sc, ind)

        nn_c0 = np.delete(nn_c0, ind, axis=1)


    Vars = list(dfI.columns)
    VarsDF=Vars.copy()
    if 'addvars' in varargs:
        Vars.extend(varargs['addvars'])
    
    # Объявляются переменные
    #b = Block(concrete=True)
    if 'name' in varargs:
        b = Block(concrete=True,name=varargs['name'])
    else:
        b = Block(concrete=True)
    b.Type='NN'
    b.State = Var(within=Binary) # Используется в add_blocks
    
    if 'block' in varargs:
        # Определяются связи между переменными текущего блока и переменными внуренних блоков
        Blocks=varargs['block']
        add_blocks(b,Blocks,Vars,t)
    else:
        Vars=unique(Vars)
        b.VarNames = Vars
        b.Vars = Var(t, Vars)
    b.t=t
    
    # Формируется структура компонента
    Values = dfs.values
    ValsF = Values[:,-1].reshape(-1,)

   
    n = nn.hidden_layer_sizes
    b.dimensions = RangeSet(0, n-1)

    b.H = Var(b.dimensions, b.t, bounds=(-10,10))
    b.aH = Var(b.dimensions, b.t, bounds=(0,10))
    b.z = Var(b.dimensions, b.t, within=Binary)
    
    def single_region_rule(b, i, t):
        Expr=0
        k=0
        #print(Vars)
        for var in VarsDF[0:-1]:
            if var in dfs.columns:
                #print(t,var,i)
                Expr += (b.Vars[t,var] - b.State*sc_mn[k])/sc_sc[k]*nn_c0[t,k,i]
                k+=1
        return  (Expr + b.State*nn_i0[t,i] == b.H[i,t])
    
    b.c_srr = Constraint(b.dimensions, b.t, rule=single_region_rule)
    
  
    def act1_rule(b,i,t):
        return b.aH[i,t] >= b.H[i,t]
    b.a1 = Constraint(b.dimensions, b.t, rule=act1_rule)
    
    def act2_rule(b,i,t):
        return b.aH[i,t] <= 10*b.z[i,t]
    b.a2 = Constraint(b.dimensions, b.t, rule=act2_rule)
    
    def act3_rule(b,i,t):
        return b.aH[i,t] <= b.H[i,t] + 10*(1-b.z[i,t]) ## 20 as (Umax-Li), but here Umax semms to be 0
    b.a3 = Constraint(b.dimensions, b.t, rule=act3_rule)
  
    # Формируем Целевую функцию
    def c_y(b, t):
        return (sum(b.aH[i,t]*nn_c1[i][0] for i in b.dimensions) + 
                b.State*nn_i1[0])*sc_sc[-1] + b.State*sc_mn[-1] == b.Vars[t,VarsDF[-1]]
    
    b.c_y = Constraint(b.t, rule = c_y) # Задаём уравнения для каждого момента t
    
    # 3. Задаётся характеристика поверхности
    # Формируется поверхность
    CH_constraints(b, dfI)
            
    # 4. Задаются ограничения типа равенство для внеших ограничений
    if  'pinned' in varargs:
        extD=varargs['pinned']
        ext_vars(b,ext=extD)    
    
    return b                 
       
def Block_Off_State(t,**varargs):
    # Block_Zero(m.t,Vars=['B','Tfw','D0'],Free_Vars=['Tfw']) 
    #b = Block(concrete=True)
    if 'name' in varargs:
        b = Block(concrete=True,name=varargs['name'])
    else:
        b = Block(concrete=True)
    b.State = Var(within=Binary) # Используется в add_blocks
    b.t=t
    b.Type='Off_State'
    b.cl=ConstraintList()
    
    Vars=varargs['Vars']
    
    if 'Free_Vars' in varargs:
        Free_Vars=set(varargs['Free_Vars'])
    else:
        Free_Vars=set()
        
    ZeroVars=set(Vars)-Free_Vars
    
    
    Vars=unique(Vars)
    b.VarNames = Vars
    MaxVal=10000
    MinVal=-10000
    b.Vars = Var(b.t, Vars,bounds=(MinVal,MaxVal))
    
    for t_ in b.t:
        for v in ZeroVars:
            b.Vars[t_,v].fix(0)
            
        for nv in Vars:#Free_Vars:
            b.cl.add(b.Vars[t_,nv]<=b.State*10000)
            b.cl.add(b.Vars[t_,nv]>=-b.State*10000)
    return b
    

def Block_Zero2(t,df,**varargs):
    # Допускаются не нулевые переменные *. Для моделирования выключенного блока 
    # CH_Block(t,df,**varargin):
    # ext - внешние ограничения
    # addvars - Список дополнительных переменных
    # 
    Values = df.values
    Vars = list(df.columns)
    ValsX = Values[:,:-1]

    #Если передаются не зануляемые переменные, то из них исключаются переменные, попавшие в ОДЗ
    if 'non_zero_vars' in varargs:
        NonZero=set(varargs['non_zero_vars'])
    else:
        NonZero=set()
        
    if 'chvars' in varargs:
        ODZ_Vars=set(varargs['chvars'])
    else:
        ODZ_Vars=set()
    ZeroVars=ODZ_Vars.union([df.columns[-1]]) #ОДЗ и значение функции долно быть 0
    
    NonZero=NonZero-ZeroVars
    

    # Добавляются вспомогательные переменные
    if 'addvars' in varargs:
        Vars.extend(varargs['addvars'])

    NonZero=NonZero.intersection(Vars)
    print('Допускаемые не нулевые переменные:', NonZero)
    # Объявляются переменные
    b = Block(concrete=True)
    b.State = Var(within=Binary) # Используется в add_blocks
    
    if 'block' in varargs:
        # Определяются связи между переменными текущего блока и переменными внуренних блоков
        Blocks=varargs['block']
        add_blocks(b,Blocks,Vars,t)
    else:
        Vars=unique(Vars)
        b.VarNames = Vars
        b.Vars = Var(t, Vars)
    b.t=t
    
    # Формируется структура компонента 
    ValsF = Values[:,-1]
    nVarDf=np.shape(Values)[1]
    

    b.c_F = ConstraintList()
    print('Обнуляемые переменные:', ZeroVars)
    for t in b.t:
        for var in ZeroVars:
            b.c_F.add(b.Vars[t,var]==0)

    
    # 4. Задаются ограничения типа равенство для внеших ограничений
    if  'pinned' in varargs:
        extD=varargs['pinned']
        ext_vars(b,ext=extD)
        
        
    # 5. Обнуление переменных при State = 0
    b.c_F5 = ConstraintList()
    for t in b.t:
        for v in set(Vars)-ZeroVars:
            if v in df.keys():
                Max=df[v].max()
                Min=df[v].min()
            else:
                Max=10000
                Min=-1000
            print('Ограничения:',v,' ',Min,' ',Max )
            b.c_F5.add(b.Vars[t,v]<=Max*b.State) 
            b.c_F5.add(b.Vars[t,v]>=Min*b.State)    
       
        
    return b   

def CH_Block(t,df,**varargs):
    # CH_Block(t,df,**varargin):
    # ext - внешние ограничения
    # addvars - Список дополнительных переменных
    # 
    Values = df.values
    Vars = list(df.columns)
    VarF = set(df.columns[-1])
    ValsX = Values[:,:-1]

    if 'no_bounds_Vars' in varargs:
        no_bounds_Vars=set(varargs['no_bounds_Vars'])
    else:
        no_bounds_Vars=set([])
      
    # Добавляются вспомогательные переменные
    if 'addvars' in varargs:
        Vars.extend(varargs['addvars'])

    # Объявляются переменные
    #b = Block(concrete=True)
    if 'name' in varargs:
        b = Block(concrete=True,name=varargs['name'])
    else:
        b = Block(concrete=True)
    b.Type='CH'
    b.State = Var(within=Binary) # Используется в add_blocks
    
    if 'block' in varargs:
        # Определяются связи между переменными текущего блока и переменными внуренних блоков
        Blocks=varargs['block']
        add_blocks(b,Blocks,Vars,t)
    else:
        Vars=unique(Vars)
        b.VarNames = Vars
        b.Vars = Var(t, Vars)
    b.t=t
    
    # Формируется структура компонента 
    ValsF = Values[:,-1]
  
    nVarDf=np.shape(Values)[1]
    
    # 2. Определяем линейную регрессию поверхности
    lm = LinearRegression()
    lm.fit(ValsX,ValsF)
    coeff = np.append(lm.coef_, np.array(-1))
    
    def c_D0_(b,t):
        expr = b.State * lm.intercept_
        k=0
        for var in Vars[0:nVarDf]:     
            expr += b.Vars[t,var]*coeff[k]
            k=k+1
            
        return expr==0
    b.c_F = Constraint(t,rule=c_D0_) 

    # 3. Формируется ОДЗ
    if 'ODZflag' in varargs:
        if varargs['ODZflag'] == False:
            pass
    else:
        if 'chvars' in varargs:
            CH_constraints(b, df[varargs['chvars']+[df.columns[-1]]])
        else:
            CH_constraints(b, df)
    
    # 4. Задаются ограничения типа равенство для внеших ограничений
    if  'pinned' in varargs:
        extD=varargs['pinned']
        ext_vars(b,ext=extD)
        
    # 5. Обнуление переменных при State = 0
    def cS0_up(b,t,v):
        if v in set(df.keys())-no_bounds_Vars:
            Max=df[v].max()
        else:
            Max=10000
        return b.Vars[t,v] <=Max*b.State
    
    def cS0_bm(b,t,v):
        if v in set(df.keys())-no_bounds_Vars:
            Min=df[v].min()
        else:
            Min=-10000
        return b.Vars[t,v] >= Min*b.State

    if 'ODZflag' in varargs:
        if varargs['ODZflag'] == False:
            pass
    else:
        b.cS0_up = Constraint(b.t, Vars, rule=cS0_up)
        b.cS0_bm = Constraint(b.t, Vars, rule=cS0_bm)
        
    return b

def PWL_Block(t,df,**varargs):
    Vars = list(df.columns)
    # Добавляются вспомогательные переменные
    if 'addvars' in varargs:
        Vars.extend(varargs['addvars'])
    
    # Объявляются переменные
    #b = Block(concrete=True)
    if 'name' in varargs:
        b = Block(concrete=True,name=varargs['name'])
    else:
        b = Block(concrete=True)
    b.Type='PWL'
    b.State = Var(within=Binary) # Используется в add_blocks
    
    if 'block' in varargs:
        # Определяются связи между переменными текущего блока и переменными внуренних блоков
        Blocks=varargs['block']
        add_blocks(b,Blocks,Vars,t)
    else:
        Vars=unique(Vars)
        b.VarNames = Vars
        b.Vars = Var(t, Vars)
    b.t=t
    
    PW=[]
    for t_ in t:
        # 1. Формируется список переменных
        # Объявляются переменные
        if  'pinned' in varargs:
            extD=varargs['pinned']
            df_=minimize_df(df,extD,t_)
        else:
            df_=df
        dfNames=df_.columns
        Values = df_.values
        ValsF = Values[:,-1].reshape(-1,)
        ValsX = Values[:,:-1]

        # Задаётся характеристка поверхности
        if len(dfNames)>2:
            # Для функции от 2-х и более переменных
            tri = Delaunay(ValsX)
            PW.append( BuildPiecewiseND_1S(b.Vars, dfNames, tri, ValsF, [t_], b.State))
        else:
            # Для функции от 1-ой переменной
            PW.append( BuildPiecewise1D_1S(b.Vars, dfNames, ValsX.reshape(-1,), ValsF, [t_], b.State))

    if  'pinned' in varargs:
        extD=varargs['pinned']
        ext_vars(b,ext=extD)
    # Добавляем всё в блок
    def add_PW(b,i):
            return PW[i].clone()
    b.PW=Block(range(0,len(PW)),rule=add_PW)
    #
    return b

# BlockStages_N
def N_Stages(t,*Blocks,**varargs):
    # Формируем ограничения по группе турбин
    # возможные дополнительные переменные:
    #
    if 'name' in varargs:
        b = Block(concrete=True,name=varargs['name'])
    else:
        b = Block(concrete=True)
    b.Type='N_Stages'
    b.t = t
    #b.q_delta=ConstraintList()
    
    # Определяем список имён переменных
    BlockVarNames=[]
    k=0
    for Bl in Blocks:
        k=k+1
        #print(k)
        for vn in Bl.VarNames:
            if vn not in BlockVarNames:
                BlockVarNames.append(vn)
                
    if 'addvars' in varargs:
        BlockVarNames.extend(varargs['addvars'])
    
    # Создаём переменные
    b.Vars = Var(b.t, BlockVarNames)
    b.VarNames=BlockVarNames
    # Переменная состояния
    b.State = Var(within=Binary)
    # Вводится переменная для управления режимом
    rk=range(k)
    b.Regime =Var(rk,within=Binary)
    
    # Сохраняем блоки в Stages
    def add_BlockStages(b,i):
        #print(i)
        return Blocks[i].clone()
    b.Stages=Block(range(0,len(Blocks)),rule=add_BlockStages)

   
    # Определяем связи между режимами оборудования
    def c_Stage(b):
        # Доработать строчку
        expr = 0
        for i in range(len(Blocks)):
            if b.Stages[i].Type!='Off_State':
              expr += b.Stages[i].State
        return expr==b.State
    b.c_stage=Constraint(rule=c_Stage)

    def c_Stage0(b):
        # Доработать строчку
        expr = 0
        for i in range(len(Blocks)):
            expr += b.Stages[i].State
        return expr<=1
    b.c_stage0=Constraint(rule=c_Stage0)  

    # Работаем с блоками
    def add_Equations_Var(b,t,VarName):
        expr = 0 
        k=0
        for i in range(len(Blocks)):
            if VarName in b.Stages[i].VarNames:
                expr += b.Stages[i].Vars[t,VarName]
                k=k+1;
        if k>0:
            return expr==b.Vars[t,VarName]
        else:
            return b.Vars[t,VarName]==b.Vars[t,VarName]
    
    b.c_equations=Constraint(b.t, BlockVarNames, rule=add_Equations_Var)

    if  'pinned' in varargs:
        extD=varargs['pinned']
        ext_vars(b,ext=extD)
        
    b.c_eq_stage=ConstraintList()
    for i in range(len(b.Stages)):
        b.c_eq_stage.add(b.Stages[i].State==b.Regime[i])
    if 'regime' in varargs:
        r=varargs['regime']
        b.c_eq_stage.add(b.Stages[r].State==b.State)
    if 'state' in varargs:
        s=varargs['state']
        b.c_eq_stage.add(b.State==s)
    return b



def CH_constraints(b, df):
    # Задаётся характеристика поверхности (ConvexHull)
    # Формируется поверхность
    Vars = list(df.columns)
    if (df.shape[1] - 1) > 1:
        hullPoints = ConvexHull(np.array(df)[:,:-1])
        planes = np.unique(hullPoints.equations,axis=0)
    elif (df.shape[1] - 1) == 1:
        planes = np.array([[-1,df.iloc[:,0].min()],[1,-df.iloc[:,0].max()]])
    else:
        raise ValueError('Zero_dimension data')
                               
    b.c_ch = ConstraintList()
    
    for t_ in b.t:
        for i in range(planes.shape[0]):
            pl = planes[i]
            Expr = 0
            k = 0
            for var in Vars[0:-1]:
                Expr += b.Vars[t_,var]*pl[k]
                k+=1
            #Expr = Expr <=(-1e-4-pl[k])*b.State[t_]
            Expr = Expr <=b.State*(-1e-4-pl[k])
            b.c_ch.add(Expr)
   
   
def add_blocks(b,Blocks,Vars,t):
    # Blocks - список блоков
    # Vars - список переменных
    # b - блок, в который добавляются переменные
    for bl in Blocks:
            VN=bl.VarNames
            Vars.extend(VN)
    # Сохраняем блоки в Block
    def add_BlockStages(b,i):
            return Blocks[i].clone()
    b.bl=Block(range(0,len(Blocks)),rule=add_BlockStages)
        # Добавляем связь между переменными
    b.c_e=ConstraintList()
    k=0
    
    Vars=unique(Vars)
    b.VarNames = Vars
    b.Vars = Var(t, Vars)
    
    print(Vars)
   
    for bl in Blocks:
            VN=bl.VarNames
            for name in VN:
                for t_ in t:
                    #print(name)
                    b.c_e.add(b.Vars[t_,name]==b.bl[k].Vars[t_,name])
            b.c_e.add(b.State==b.bl[k].State)    
            k=k+1
    return b
                
def ext_vars(b,**ext):
    Vars=b.VarNames
    if len(ext)>0: # Если словарь задан
        #extD=ext[0]   # Словарь
        # Проверяем: есть ли переменные в Vars
        # Определяем функцию формирования ограничений
        if 'ext' in ext:
            ext=ext['ext']
        def c_Ext(b,t,ExtVar):
            #b.Vars[t,var] == ext_df.loc[t,var]
            if b.State.dim()==0:
                State=b.State
            else:
                State=b.State[t]
            if isinstance(ext[ExtVar], numbers.Number):
                return b.Vars[t,ExtVar] == State*ext[ExtVar]#[t]
            elif len(ext[ExtVar])>=t: 
                return b.Vars[t,ExtVar] == State*ext[ExtVar][t]#[t]
            else:
                return b.Vars[t,ExtVar] == State*ext[ExtVar][0]#[t]
        # Проверяем корректность написания ограничений и добавляем в массив
        Ext = []

        for n in Vars:
            if n in ext:   
                Ext.append(n)
        # формируем набор ограничений для каждого элемента
        b.c_ExtConst = Constraint(b.t, Ext, rule = c_Ext)
    return b
    
    
    
    
def BuildPiecewiseND_1S(vars_, namevars, tri, zvals,t1,State):
    """
    Builds constraints defining a N-dimensional
    piecewise representation of the given triangulation.

    Args:
        xvars: A (D, 1) array of Pyomo variable objects
               representing the inputs of the piecewise
               function.
        zvar: A Pyomo variable object set equal to the
              output of the piecewise function.
        tri: A triangulation over the discretized
             variable domain. Required attributes:
           - points: An (npoints, D) shaped array listing the
                     D-dimensional coordinates of the
                     discretization points.
           - simplices: An (nsimplices, D+1) shaped array of
                        integers specifying the D+1 indices
                        of the points vector that define
                        each simplex of the triangulation.
        zvals: An (npoints, 1) shaped array listing the
               value of the piecewise function at each of
               coordinates in the triangulation points
               array.

    Returns:
        A Pyomo Block object containing variables and
        constraints that define the piecewise function.
    """

    b = Block(concrete=True)
    nt=len(t1)
    ndim = len(namevars)-1
    nsimplices = len(tri.simplices)
    npoints = len(tri.points)
    pointsT = list(zip(*tri.points))

    # create index objects
    # Список переменных [X Y Z]
    b.dimensions = RangeSet(0, ndim-1)
    # Списое полигонов
    b.simplices = RangeSet(0, nsimplices-1)
    # Точки
    b.vertices = RangeSet(0, npoints-1)
    # 
    b.t=RangeSet(0,nt-1)
    
    # create variables
    # Лямбда
    b.lmda = Var(b.vertices,b.t, within=NonNegativeReals)
    # y - Принадлежность полигону
    b.y = Var(b.simplices,b.t, within=Binary)

    # create constraints
    # X=сумма(X[i]*Лямбда[i])
    # Y=сумма(Y[i]*Лямбда[i])
    # Z=сумма(Z[i]*Лямбда[i])
    # Для упрощения конструкции
    # d=[X Y Z]

    b.input_c1=ConstraintList()
    for d in b.dimensions:
        for t in b.t:
            pointsTd = pointsT[d]
            b.input_c1.add(vars_[t1[t],namevars[d]] == sum(pointsTd[v]*b.lmda[v,t]
                               for v in b.vertices))
    
    # F=сумма(F[i]*Лямбда[i])
    for t in b.t:
        b.input_c1.add(vars_[t1[t],namevars[-1]] == sum(zvals[v]*b.lmda[v,t] for v in b.vertices))

    
    #сумма(Лямбда[i])==1
    for t in b.t:
        b.input_c1.add(sum(b.lmda[v,t] for v in b.vertices) == State)
            
    # generate a map from vertex index to simplex index,
    # which avoids an n^2 lookup when generating the
    # constraint
    
    # Формируем карту принадлежности полигону:
    # Лямбда[i]<=сумм(y(смежные с точкой полигоны))
    vertex_to_simplex = [[] for v in b.vertices]
    for s, simplex in enumerate(tri.simplices):
        for v in simplex:
            vertex_to_simplex[v].append(s)
            
    def vertex_regions_rule(b, v,t):
        return b.lmda[v,t] <= \
            sum(b.y[s,t] for s in vertex_to_simplex[v])
    b.vertex_regions_c = \
        Constraint(b.vertices,b.t, rule=vertex_regions_rule)
    
    # Всегда выбираем один из полигонов:
    # Сумм(полигон[i])==1
    def single_region_rule(b,t):
        return sum(b.y[s,t] for s in b.simplices) == State
    b.single_region_c = Constraint(b.t,rule=single_region_rule)
    #b.single_region_c = Constraint(expr=\
    #    sum(b.y[s] for s in b.simplices) == 1)

    return b

def BuildPiecewise1D_1S(var_,namevar,x,zvals,t1,State):
    
    # Одномерная характеристика
    b = Block(concrete=True)
    npoints = len(zvals)
    nt=len(t1)
    b.vertices = RangeSet(0, npoints-1)
    b.simplices = RangeSet(0, npoints-2)
    b.t=RangeSet(0,nt-1)
    
    b.lmda = Var(b.vertices,b.t, within=NonNegativeReals)
    b.y = Var(b.simplices,b.t, within=Binary)
    
    b.input_c1=ConstraintList()
    # сумма лябмд = State
    for t in b.t:
        b.input_c1.add(sum(b.lmda[v,t] for v in b.vertices) == State)
  
    # F=сумма(zi*лямбда i) 
    for t in b.t:
        b.input_c1.add(var_[t1[t],namevar[-1]] == sum(zvals[v]*b.lmda[v,t] for v in b.vertices))
   
    # x=сумма(xi*лямбда i)
    for t in b.t:
        b.input_c1.add(var_[t1[t],namevar[0]] == sum(x[v]*b.lmda[v,t] for v in b.vertices))
    
    # Только две соседние лямбды не нулевые
    vertex_to_simplex = [[] for v in b.vertices]
    
    simplices = [[v,v+1] for v in b.vertices]
    simplices=simplices[:-1]
    
    for s, simplex in enumerate(simplices):
        for v in simplex:
            vertex_to_simplex[v].append(s)
    #print(vertex_to_simplex)
    
            
    def vertex_regions_rule(b,v,t):
        return  b.lmda[v,t] <=  sum(b.y[s,t] for s in vertex_to_simplex[v])
    b.vertex_regions_c =Constraint(b.vertices, b.t, rule=vertex_regions_rule)
    
    # Всегда выбираем один из полигонов:
    # Сумм(полигон[i])==1
    def single_region_rule(b,t):
        return sum(b.y[s,t] for s in b.simplices) == State
    b.single_region_c = Constraint(b.t,rule=single_region_rule)
    return b
    
            
    def vertex_regions_rule(b,v,t):
        return  b.lmda[v,t] <=  sum(b.y[s,t] for s in vertex_to_simplex[v])
    b.vertex_regions_c =Constraint(b.vertices, b.t, rule=vertex_regions_rule)
    
    # Всегда выбираем один из полигонов:
    # Сумм(полигон[i])==1
    def single_region_rule(b,t):
        return sum(b.y[s,t] for s in b.simplices) == State[t]
    b.single_region_c = Constraint(b.t,rule=single_region_rule)
    return b
    
def unique(Names):
    out=[]
    for n in Names:
        if n not in out:
            out.append(n)
    return out

# Отображение переменных
def disp(block):
    dct = dict(block.Vars.get_values())
    tdf = pd.DataFrame.from_dict(dct, orient="index")
    tdf.reset_index(inplace=True)
    tdf['hour'] = tdf['index'].apply(lambda s: s[0])
    tdf['var'] = tdf['index'].apply(lambda s: s[1])
    tdf = tdf.set_index(['hour','var']).drop('index',axis=1).unstack()
    tdf.columns = tdf.columns.droplevel()
    
    return tdf

# Modeling
def nn2df(nn,scaler,ch_df, max_tol=0.01, mean_tol=0.005):
    
    combs = combinations(range(nn.hidden_layer_sizes),nn.coefs_[0].shape[0])
    scales = scaler.scale_[:-1]
    means = scaler.mean_[:-1]
    
    if nn.coefs_[0].shape[0]>1:
        ch = ConvexHull(ch_df.iloc[:,:-1])
        planes = ch.equations
    elif nn.coefs_[0].shape[0]==1:
        planes = np.array([[-1,ch_df.iloc[:,0].min()],[1,-ch_df.iloc[:,0].max()]])

    N=ch_df.shape[0]

    df_list = []
    for comb in combs:
        a = nn.coefs_[0][:,comb]
        b = nn.intercepts_[0][[i for i in comb]]
        x = np.linalg.solve(a,b).reshape(1,-1)
        X = x*scales+means
        X1 = np.concatenate([X,np.ones((1,1))],axis=1)

        if np.max(X1@planes.T) < 0:

            x1 = np.concatenate([x,nn.predict(x).reshape(1,-1)],axis=1)
            X1 = scaler.inverse_transform(x1)
            kdf = pd.DataFrame(data = X1,columns=ch_df.columns,index=[N])
            df_list.append(kdf)
            N+=1
    
    res_df = pd.concat([ch_df]+df_list)
       
    max_acc = np.inf
    mean_acc = np.inf
    ind = list(ConvexHull(dfT1_1).vertices)
    X_sc = res_df.iloc[:,:-1]
    y_sc = res_df.iloc[:,-1]
    N=0
    
    if nn.coefs_[0].shape[0]==1:
        return res_df
    
    while (max_acc>max_tol)|(mean_acc>mean_tol):
        tri = Delaunay(X_sc.loc[ind])
        f = LinearNDInterpolator(tri,y_sc.loc[ind])
        not_ind = list(set(range(X_sc.shape[0])).difference(set(ind)))
        predict = f(X_sc.loc[not_ind])
        res = np.abs(y_sc.loc[not_ind]-predict)
        worst = res.idxmax()
        max_acc = res.max()/y_sc.loc[not_ind].mean()
        mean_acc = res.mean()/y_sc.loc[not_ind].mean()
        ind = ind+[worst]
        print(N,max_acc,mean_acc)
        N+=1
            
    return res_df.iloc[ind]

def ch_reduce(df,scaler,tol):
    df_small = pd.DataFrame(data=scaler.transform(df),index = df.index)
    s0 = ConvexHull(df_small.iloc[:,:-1]).area
    s = s0

    while s>tol*s0:
        res = {i: s - ConvexHull(df_small.drop(i).iloc[:,:-1]).area for i in df_small.index}
        res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1])}
        df_small.drop(list(res.items())[0][0], inplace=True)
        s = ConvexHull(df_small.iloc[:,:-1]).area
        print(df_small.shape[0], s/s0)

    return pd.DataFrame(scaler.inverse_transform(df_small), columns=df.columns)

"""
def extend_ch(ch_df, params = {'N': {'down':0, 'up': 0}}):
    
    lm = LinearRegression()
    lm.fit(ch_df.iloc[:,:-1],ch_df.iloc[:,-1])
    
    ext_df = ch_df.iloc[:,:-1].copy()
    plus_df = ch_df.iloc[:,:-1].copy()
    mins_df = ch_df.iloc[:,:-1].copy()

    for col in params:
        plus_df[col] = plus_df[col] + params[col]['up']
        mins_df[col] = mins_df[col] - params[col]['down']

    ext_df = ext_df.append([plus_df,mins_df])
    
    if ch_df.shape[1] > 2:

        ch = ConvexHull(ext_df)
        ext_df = ext_df.iloc[ch.vertices]

        
    ext_df['D0'] = lm.predict(ext_df)
    
    return ext_df.drop_duplicates().reset_index(drop=True)
"""
opt = SolverFactory('scip')

def check_CH(point, df, chvars):
    
    ch = ConvexHull(df[chvars])
    eq = ch.equations.T
    dist = point@eq[:-1] + eq[-1]
    over_index = np.where(dist > 0)
    over_eq = ch.equations[over_index]
    over_dist = dist[over_index]

    
    if len(over_dist) > 0:
        
        min_positive = np.argmin(over_dist)
        over_eq = over_eq
        
        m = ConcreteModel()

        x0 = {chvars[c]: point[c] for c in range(len(chvars))}
        m.x = Var(chvars)

        m.c_ch = ConstraintList()
        planes = ch.equations

        for i in range(planes.shape[0]):
            pl = planes[i]
            Expr = 0
            k = 0
            for var in chvars:
                Expr += m.x[var]*pl[k]
                k+=1
            Expr = Expr + pl[k] <= 0
            m.c_ch.add(Expr)

        m.dist = Objective(expr= sum((x0[c]-m.x[c])**2 for c in chvars))

        status = opt.solve(m)
        lm = LinearRegression().fit(df.iloc[:,:-1], df.iloc[:,-1])
        coefs = dict(zip(lm.feature_names_in_, lm.coef_))
        d0_dist = np.sqrt(sum(((x0[c]-m.x[c].value)*coefs[c])**2 for c in chvars))
        
        return d0_dist, {c: x0[c] - m.x[c].value for c in chvars}
    
    else:
        
        return 0, {}
    
def extend_ch(ch_df, params = {'N': 0}):
    
    lm = LinearRegression()
    lm.fit(ch_df.iloc[:,:-1],ch_df.iloc[:,-1])
    
    ext_df = ch_df.iloc[:,:-1].copy()
    add_df = ch_df.iloc[:,:-1].copy()

    for col in params:
        add_df[col] = add_df[col] + params[col]

    ext_df = ext_df.append([add_df])
    
    if ch_df.shape[1] > 2:

        ch = ConvexHull(ext_df)
        ext_df = ext_df.iloc[ch.vertices]

        
    ext_df[ch_df.columns[-1]] = lm.predict(ext_df)
    
    return ext_df.drop_duplicates().reset_index(drop=True)
    
def add_curve(f,Name,X,F):
    if type(f)!=dict:
        f={}
        
    n=np.shape(X);
    if len(n)==1:
        f.update({Name:interp1d(X,F,bounds_error=False, fill_value='extrapolate')})
    else:
        f.update({Name:LinearNDInterpolator(X, F,rescale=True)})      
    return f
    
    
def simplify(df,dQmax):
    # Блок прореживает характеристики. dQmax - максимальная допустимая погрешность в единицах
    Columns=df.columns
    used_P=[];
    if len(Columns)>3:
        for i in np.unique(df[Columns[0]]):
            temp=df[df[Columns[0]]==i]
            hullPoints = ConvexHull(np.array(temp)[:,1:-1])
            used_P.extend(list(temp.iloc[hullPoints.vertices].index))
    else:
        hullPoints = ConvexHull(np.array(df)[:,:-1])
        used_P=hullPoints.vertices

    print(used_P)
    Error=[10]
    index=0
    CNames=df.columns
    while Error[index]>dQmax:
        # Определяем выпуклое множество точек
        df_l=df.loc[used_P]
        f = add_curve({},'Q',df_l[CNames[0:-1]],df_l[CNames[-1]])
        # Считаем максимальную погрешность
        Error=abs(f['Q'](df[CNames[0:-1]]).ravel()-df[CNames[-1]])
        index=np.nanargmax(Error)
        print(Error[index])
        used_P=np.append(used_P,index)
    used_P=np.sort(used_P)
    result=df.loc[used_P].reset_index(drop=True)
    return  result

def minimize_df(df,extD,t=0):
    f = LinearNDInterpolator(df.iloc[:,:-1], df.iloc[:,-1])
    new_df = df.copy()
    #add2ext=[]
    for v in extD.keys():
        #add2ext.append(v)
        values=extD[v]
        if isinstance(values, numbers.Number):
                value=values
        elif len(values)>=t: 
                value=values[t]
        else:
                value=values[0]
        new_df[v] = value
        new_df.iloc[:,-1] = f(new_df.iloc[:,:-1])
    df = new_df.drop(extD.keys(), axis=1).drop_duplicates()
    return df#,add2ext

def list_constraint(m):
    Vars=[]
    for Var1 in m.component_data_objects(Constraint):
        Vars.append(Var1.name)
    return Vars   
    
def list_ovjective(m):
    Vars=[]
    for Var1 in m.component_data_objects(Objective):
        Vars.append(Var1.name)
    return Vars
    
def list_vars(m):
    Vars=[]
    for Var1 in m.component_data_objects(Var):
        Vars.append(Var1.name)
    return Vars    

