from pyomo.environ import *
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
    
    
def NN_Block(t,dfI,nn,scaler,*ext):
    # nn     # Структура нейронной сети
    # ext    # Ограничения на переменные
    # dfI    # Датафрейм для построения ограничивающей выпуклой оболочки

    Vars = dfI.columns
    
    b = Block(concrete=True)
    b.t = t
    Values = dfI.values
    ValsF = Values[:,-1].reshape(-1,)

    # 1. Формируется список переменных
    # Объявляются переменные
    b.Vars = Var(b.t, Vars)
    b.Stage = Var(within=Binary)
    b.VarNames = Vars
    
    n = nn.hidden_layer_sizes
    b.dimensions = RangeSet(0, n-1)

    b.H = Var(b.dimensions, b.t, bounds=(-10000,10000))
    b.aH = Var(b.dimensions, b.t, bounds=(-10000,10000))
    
    def single_region_rule(b, i, t):
        Expr=0
        k=0
        for var in Vars[0:-1]:
            Expr += (b.Vars[t,var] - b.Stage*scaler.mean_[k])/scaler.scale_[k]*nn.coefs_[0][k,i]
            k+=1
        return  (Expr + b.Stage*nn.intercepts_[0][i] == b.H[i,t])
    
    b.c_srr = Constraint(b.dimensions, b.t, rule=single_region_rule)
    
    if nn.activation == 'relu':
        xdata = [-10000., 0., 10000.]
        ydata = [   0., 0., 10000.]
    elif nn.activation == 'identity':
        xdata = [-10000., 0., 10000.]
        ydata = [-10000., 0., 10000.]
    else:
        raise ValueError('Only relu and identity activation fuctions are allowed for neural network')
        
    #Фунция ReLU: aH = max(0,H) or aH==H in 'identity' case
            
    b.Cons = Piecewise(b.dimensions, b.t, b.aH, b.H, pw_pts=xdata, pw_constr_type="EQ", f_rule=ydata, pw_repn="SOS2")
    
    # Формируем Целевую функцию
    def c_y(b, t):
        return (sum(b.aH[i,t]*nn.coefs_[1][i][0] for i in b.dimensions) + 
                b.Stage*nn.intercepts_[1][0])*scaler.scale_[-1] + b.Stage*scaler.mean_[-1] == b.Vars[t,Vars[-1]]
    
    b.c_y = Constraint(b.t, rule = c_y) # Задаём уравнения для каждого момента t
    
    # 3. Задаётся характеристика поверхности
    # Формируется поверхность
    if nn.coefs_[0].shape[0]>1:
        hullPoints = ConvexHull(np.array(dfI)[:,:-1])
        planes = np.unique(hullPoints.equations,axis=0)
    elif nn.coefs_[0].shape[0]==1:
        planes = np.array([[-1,dfI.iloc[:,0].min()],[1,-dfI.iloc[:,0].max()]])
    else:
        raise ValueError('Zero_dimension data')
                               
    b.c_ch = ConstraintList()
    for t_ in b.t:
        for i in range(planes.shape[0]):
                pl=planes[i]
                Expr=0
                k=0
                for var in Vars[0:-1]:
                    Expr += b.Vars[t_,var]*pl[k]
                    k+=1
                Expr = Expr <= b.Stage*(-1e-4-pl[k])
                b.c_ch.add(Expr)
                
    # 4. Задаём ограничения на переменные

    if len(ext)>0:
        extD=ext[0]   # Словарь
        # Проверяем: есть ли переменные в Vars
        # Определяем функцию формирования ограничений
        def c_Ext(b,t,ExtVar):
            return b.Vars[t,ExtVar] == b.Stage*extD[ExtVar]
        # Проверяем корректность написания ограничений и добавляем в массив
        Ext = []
        for n in Vars:
            if n in extD:   
                Ext.append(n)
        # формируем набор ограничений для каждого элемента
        b.c_ExtConst = Constraint(b.t, Ext, rule = c_Ext)
        
    return b


def CH_Block(t,df,*ext):
    
    Values = df.values
    Vars = list(df.columns)

    b = Block(concrete=True) 
    
    ValsX = Values[:,:-1]
    ValsF = Values[:,-1]

    # 1. Формируется список переменных
    # Объявляются переменные
    b.Vars = Var(t, Vars)
    b.Stage = Var(within=Binary)
    b.VarNames = Vars
    b.t=t
    # 2. Определяем линейную регрессию поверхности

    lm = LinearRegression()
    lm.fit(ValsX,ValsF)
    coeff = np.append(lm.coef_, np.array(-1))
    
    def c_D0_(b,t):

        expr = b.Stage * lm.intercept_
        k=0
        for var in Vars:     
            expr += b.Vars[t,var]*coeff[k]
            k=k+1
            
        return expr==0
    
    b.c_F = Constraint(t,rule=c_D0_) 

    # 3. Задаётся характеристка поверхности
    # Формируется поверхность
    if len(Vars)>2:
        hullPoints = ConvexHull(np.array(df)[:,:-1])
        planes = np.unique(hullPoints.equations,axis=0)
    elif len(Vars)==2:
        planes = np.array([[-1,ValsX[:,0].min()],[1,-ValsX[:,0].max()]])
    else:
        raise ValueError('Zero_dimension data')

    
    b.c_ch = ConstraintList()
    for t_ in t:
        for i in range(planes.shape[0]):
            pl=planes[i]
            Expr=0
            k=0
            for var in Vars[0:-1]:
                Expr += b.Vars[t_,var]*pl[k]
                k+=1
            Expr = Expr <= b.Stage*(-1e-4-pl[k])
            b.c_ch.add(Expr)
                
    if len(ext)>0:
        extD=ext[0]   # Словарь
        # Проверяем: есть ли переменные в Vars
        # Определяем функцию формирования ограничений
        def c_Ext(b,t,ExtVar):
            return b.Vars[t,ExtVar] == b.Stage*extD[ExtVar]
        # Проверяем корректность написания ограничений и добавляем в массив
        Ext = []
        for n in Vars:
            if n in extD:   
                Ext.append(n)
        # формируем набор ограничений для каждого элемента
        b.c_ExtConst = Constraint(t, Ext, rule = c_Ext)
        
    return b

def BuildPiecewise1D_1S(var_,namevar,x,zvals,t,stage):
    
    # Одномерная характеристика
    b = Block(concrete=True)
    npoints = len(zvals)
    nt=len(t)
    b.vertices = RangeSet(0, npoints-1)
    b.simplices = RangeSet(0, npoints-2)
    b.t=RangeSet(0,nt-1)
    
    b.lmda = Var(b.vertices,b.t, within=NonNegativeReals)
    b.y = Var(b.simplices,b.t, within=Binary)
    
    # сумма лябмд = stage
    def lmbda_summ_rule(b,t):
        return sum(b.lmda[v,t] for v in b.vertices) == stage
    b.convex_c=Constraint(b.t, rule=lmbda_summ_rule)
    
    # z=сумма(zi*лямбда i) 
    def z_var_rule(b,t):
        return var_[t,namevar[-1]] == sum(zvals[v]*b.lmda[v,t] for v in b.vertices)
    b.output_c=Constraint(b.t, rule=z_var_rule)
    
    # x=сумма(xi*лямбда i)
    def x_var_rule(b,t):
        return var_[t,namevar[0]] == sum(x[v]*b.lmda[v,t] for v in b.vertices)
    b.inpuit_c=Constraint(b.t, rule=x_var_rule)
    
    
    # Только две соседние лямбды не нулевые
    vertex_to_simplex = [[] for v in b.vertices]
    #vertex_to_simplex=vertex_to_simplex[:-1]
    simplices = [[v,v+1] for v in b.vertices]
    simplices=simplices[:-1]
    #print(simplices)
    #vertex_to_simplex[0]=b.vertices.first()
    #vertex_to_simplex[-1]=b.vertices.last()
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
        return sum(b.y[s,t] for s in b.simplices) == stage
    b.single_region_c = Constraint(b.t,rule=single_region_rule)
    return b
    
            
    def vertex_regions_rule(b,v,t):
        return  b.lmda[v,t] <=  sum(b.y[s,t] for s in vertex_to_simplex[v])
    b.vertex_regions_c =Constraint(b.vertices, b.t, rule=vertex_regions_rule)
    
    # Всегда выбираем один из полигонов:
    # Сумм(полигон[i])==1
    def single_region_rule(b,t):
        return sum(b.y[s,t] for s in b.simplices) == stage[t]
    b.single_region_c = Constraint(b.t,rule=single_region_rule)
    return b

def BuildPiecewiseND_1S(vars_, namevars, tri, zvals,t,stage):
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
    nt=len(t)
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
    def input_c_rule(b, d,t):
        #print(d)
        #breakpoint()
        pointsTd = pointsT[d]
        return vars_[t,namevars[d]] == sum(pointsTd[v]*b.lmda[v,t]
                               for v in b.vertices)
    b.input_c = Constraint(b.dimensions, b.t, rule=input_c_rule)
    
    # F=сумма(F[i]*Лямбда[i])
    def z_var_rule(b,t):
        return vars_[t,namevars[-1]] == sum(zvals[v]*b.lmda[v,t] for v in b.vertices)
    b.output_c=Constraint(b.t, rule=z_var_rule)
    #b.output_c = Constraint(expr=\
    #    zvar == sum(zvals[v]*b.lmda[v] for v in b.vertices))
    
    #сумма(Лямбда[i])==1
    def lmbda_summ_rule(b,t):
        return sum(b.lmda[v,t] for v in b.vertices) == stage
    b.convex_c=Constraint(b.t, rule=lmbda_summ_rule)
    #b.convex_c = Constraint(expr=\
    #    sum(b.lmda[v] for v in b.vertices) == 1)

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
        return sum(b.y[s,t] for s in b.simplices) == stage
    b.single_region_c = Constraint(b.t,rule=single_region_rule)
    #b.single_region_c = Constraint(expr=\
    #    sum(b.y[s] for s in b.simplices) == 1)

    return b

def PWL_Block(t,df):
    
    Values = df.values
    VarNames = df.columns

    
    ValsF = Values[:,-1].reshape(-1,)
    ValsX = Values[:,:-1]

    # 1. Формируется список переменных
    # Объявляются переменные
    b = Block(concrete=True)
    b.t = t
    b.Vars = Var(b.t,VarNames)
    b.Stage = Var(within=Binary)
    b.VarNames = VarNames
   
    # 2. Задаётся характеристка поаерхности
    if len(VarNames)>2:
        # Для функции от 2-х и более переменных
        tri = Delaunay(ValsX)
        b.PW = BuildPiecewiseND_1S(b.Vars, VarNames, tri, ValsF, b.t, b.Stage)
    else:
        # Для функции от 1-ой переменной
        b.PW = BuildPiecewise1D_1S(b.Vars, VarNames, ValsX.reshape(-1,), ValsF, b.t, b.Stage)
        
    return b

def BlockStages_N(t,*Blocks):
    # Формируем ограничения по группе турбин
    b = Block(concrete=True)
    b.t = t
    
    # Определяем список имён переменных
    BlockVarNames=[]
    
    for Bl in Blocks:
        for vn in Bl.VarNames:
            if vn not in BlockVarNames:
                BlockVarNames.append(vn)
    
    # Создаём переменные
    b.Vars = Var(b.t, BlockVarNames)
    # Переменная состояния
    b.Stage = Var(within=Binary) 
   
    
    # Сохраняем блоки в Block
    def add_BlockStages(b,i):
        return Blocks[i].clone()
    b.Stages=Block(range(0,len(Blocks)),rule=add_BlockStages)

   
    # Определяем связи между режимами оборудования
    def c_Stage(b):
        # Доработать строчку
        expr = 0
        for i in range(len(Blocks)):
            expr += b.Stages[i].Stage
        return expr==b.Stage
    b.c_stage=Constraint(rule=c_Stage)

    # Работаем с блоками
    def add_Equations_Var(b,t,VarName):
        expr = 0 
        for i in range(len(Blocks)):
            if VarName in b.Stages[i].VarNames:
                expr += b.Stages[i].Vars[t,VarName]
        return expr==b.Vars[t,VarName]
    
    b.c_equations=Constraint(b.t, BlockVarNames, rule=add_Equations_Var)
    return b

def CH_constraints(b, df):
    # Задаётся характеристика поверхности (ConvexHull)
    # Формируется поверхность
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
            Expr = Expr <=(-1e-4-pl[k])*b.Stage[t_]
            b.c_ch.add(Expr)
                
def ext_vars(b,**ext):
    Vars=b.VarNames
    if len(ext)>0: # Если словарь задан
        #extD=ext[0]   # Словарь
        # Проверяем: есть ли переменные в Vars
        # Определяем функцию формирования ограничений
        def c_Ext(b,t,ExtVar):
            if b.Stage.dim()==0:
                Stage=b.Stage
            else:
                Stage=b.Stage[t]
            if isinstance(ext[ExtVar], numbers.Number):
                return b.Vars[t,ExtVar] == Stage*ext[ExtVar]#[t]
            elif len(ext[ExtVar])>=t: 
                return b.Vars[t,ExtVar] == Stage*ext[ExtVar][t]#[t]
            else:
                return b.Vars[t,ExtVar] == Stage*ext[ExtVar][0]#[t]
        # Проверяем корректность написания ограничений и добавляем в массив
        Ext = []
        #print(Vars)
        #print(ext)
        for n in Vars:
            if n in ext:   
                Ext.append(n)
        # формируем набор ограничений для каждого элемента
        print(Ext)
        b.c_ExtConst = Constraint(b.t, Ext, rule = c_Ext)
    return b

def disp(block):
    dct = dict(block.Vars.get_values())
    tdf = pd.DataFrame.from_dict(dct, orient="index")
    tdf.reset_index(inplace=True)
    tdf['hour'] = tdf['index'].apply(lambda s: s[0])
    tdf['var'] = tdf['index'].apply(lambda s: s[1])
    tdf = tdf.set_index(['hour','var']).drop('index',axis=1).unstack()
    tdf.columns = tdf.columns.droplevel()
    
    return tdf


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