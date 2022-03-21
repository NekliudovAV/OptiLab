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
            b.q_delta.add(b.dVar[t_,dVarname[0]] <= N_min * (1 - b.Stage) + b.Stage * dN)
            b.q_delta.add(b.dVar[t_,dVarname[1]] <= N_min * (1 - b.Stage) + b.Stage * dN)
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

    Vars = dfI.columns
    if 'addvars' in varargs:
        Vars.extend(varargs['addvars'])
    

    b = Block(concrete=True)
    b.t = t
    Values = dfs.values
    ValsF = Values[:,-1].reshape(-1,)

    # 1. Формируется список переменных
    # Объявляются переменные
    b.Vars = Var(b.t, Vars)
    b.Stage = Var(within=Binary)
    b.VarNames = Vars
    
    n = nn.hidden_layer_sizes
    b.dimensions = RangeSet(0, n-1)

    b.H = Var(b.dimensions, b.t, bounds=(-10,10))
    b.aH = Var(b.dimensions, b.t, bounds=(0,10))
    b.z = Var(b.dimensions, b.t, within=Binary)
    
    def single_region_rule(b, i, t):
        Expr=0
        k=0
        for var in Vars[0:-1]:
            if var in dfs.columns:
              Expr += (b.Vars[t,var] - b.Stage*sc_mn[k])/sc_sc[k]*nn_c0[t,k,i]
              k+=1
        return  (Expr + b.Stage*nn_i0[t,i] == b.H[i,t])
    
    b.c_srr = Constraint(b.dimensions, b.t, rule=single_region_rule)
    
    def pinned_rule(b, var, t):
      return b.Vars[t,var] == ext_df.loc[t,var]
    b.pinned = Constraint(ext_df.columns, b.t, rule=pinned_rule)

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
                b.Stage*nn_i1[0])*sc_sc[-1] + b.Stage*sc_mn[-1] == b.Vars[t,Vars[-1]]
    
    b.c_y = Constraint(b.t, rule = c_y) # Задаём уравнения для каждого момента t
    
    # 3. Задаётся характеристика поверхности
    # Формируется поверхность
    if nn_c0.shape[1]>1:
        hullPoints = ConvexHull(np.array(dfs)[:,:-1])
        planes = np.unique(hullPoints.equations,axis=0)
        
    elif nn_c0.shape[1]==1:
        planes = np.array([[-1,dfs.iloc[:,0].min()],[1,-dfs.iloc[:,0].max()]])
    else:
        raise ValueError('Zero_dimension data')
                               
    b.c_ch = ConstraintList()
    for t_ in b.t:
        for i in range(planes.shape[0]):
            pl=planes[i]
            Expr=0
            k=0
            for var in Vars[0:-1]:
              if var in dfs.columns:
                Expr += b.Vars[t_,var]*pl[k]
                k+=1
            Expr = Expr <= b.Stage*(-1e-4-pl[k])
            b.c_ch.add(Expr)
            
    if 'block' in varargs:
        Blocks=varargs['block']
        add_blocks(b,Blocks)
        
    return b          
       


def CH_Block(t,df,**varargs):
    # CH_Block(t,df,**varargin):
    # ext - внешние ограничения
    # addvars - Список дополнительных переменных
    
    # 
    Values = df.values
    Vars = list(df.columns)
    b = Block(concrete=True)
    b.q_delta=ConstraintList()
    
    dfNames = Vars.copy()
  
    
    if 'addvars' in varargs:
        Vars.extend(varargs['addvars'])
          
    Vars=unique(Vars)
    ValsX = Values[:,:-1]
    ValsF = Values[:,-1]
    nVarDf=np.shape(Values)[1]
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
        for var in Vars[0:nVarDf]:     
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
            for var in Vars[0:nVarDf-1]:
                Expr += b.Vars[t_,var]*pl[k]
                k+=1
            Expr = Expr <= b.Stage*(-1e-4-pl[k])
            b.c_ch.add(Expr)
            
    if  'pinned' in varargs:
        extD=varargs['pinned']
        ext_vars(b,ext=extD)
        
    if 'block' in varargs:
        Blocks=varargs['block']
        add_blocks(b,Blocks)
    return b






def PWL_Block(t,df,**varargs):
    # 1. Формируется список переменных
    # Объявляются переменные
    b = Block(concrete=True)
    
    Values = df.values
    VarNames = list(df.columns)
    dfNames = VarNames.copy()
    if 'block' in varargs:
        VN=varargs['block'].VarNames
        VarNames.extend(VN)
        b.bl=varargs['block'].clone()
        
    if 'addvars' in varargs:
        VarNames.extend(varargs['addvars'])
    b.q_delta=ConstraintList()
    VarNames=unique(VarNames)
    
    ValsF = Values[:,-1].reshape(-1,)
    ValsX = Values[:,:-1]

   
    b.t = t
    b.Vars = Var(b.t,VarNames)
    b.Stage = Var(within=Binary)
    b.VarNames = VarNames
   
    # 2. Задаётся характеристка поаерхности
    if len(VarNames)>2:
        # Для функции от 2-х и более переменных
        tri = Delaunay(ValsX)
        b.PW = BuildPiecewiseND_1S(b.Vars, dfNames, tri, ValsF, b.t, b.Stage)
    else:
        # Для функции от 1-ой переменной
        b.PW = BuildPiecewise1D_1S(b.Vars, dfNames, ValsX.reshape(-1,), ValsF, b.t, b.Stage)
    if  'pinned' in varargs:
        extD=varargs['pinned']
        ext_vars(b,ext=extD)
    
    if 'block' in varargs:
        # Добавляем связь между переменными
        b.c_e=ConstraintList()
        for name in VN:
            for t_ in t:
                b.c_e.add(b.Vars[t,name]==b.bl.Vars[t,name])
        b.c_e.add(b.Stage==b.bl.Stage)
        
    return b

# BlockStages_N
def N_Stages(t,*Blocks,**varargs):
    # Формируем ограничения по группе турбин
    # возможные дополнительные переменные:
    #
    b = Block(concrete=True)
    b.t = t
    b.q_delta=ConstraintList()
    
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
    b.Stage = Var(within=Binary)
    # Вводится переменная для управления режимом
    rk=range(k)
    b.Regime =Var(rk,within=Binary)
   
    
    # Сохраняем блоки в Block
    def add_BlockStages(b,i):
        #print(i)
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
        b.c_eq_stage.add(b.Stages[i].Stage==b.Regime[i])
    if 'regime' in varargs:
        r=varargs['regime']
        b.c_eq_stage.add(b.Stages[r].Stage==b.Stage)
    if 'stage' in varargs:
        s=varargs['stage']
        b.c_eq_stage.add(b.Stage==s)
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
   
   
def add_blocks(b,Blocks_):
        for bl in Blocks_:
            VN=bl.VarNames
            Vars.extend(VN)
        # Сохраняем блоки в Block
        def add_BlockStages(b,i):
            return Blocks[i].clone()
        b.bl=Block(range(0,len(Blocks)),rule=add_BlockStages)
        # Добавляем связь между переменными
        b.c_e=ConstraintList()
        k=0
        print(Vars)
        for bl in Blocks_:
            VN=bl.VarNames
            for name in VN:
                for t_ in b.t:
                    print(name)
                    b.c_e.add(b.Vars[t_,name]==b.bl[k].Vars[t_,name])
            b.c_e.add(b.Stage==b.bl[k].Stage)    
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

        for n in Vars:
            if n in ext:   
                Ext.append(n)
        # формируем набор ограничений для каждого элемента
        b.c_ExtConst = Constraint(b.t, Ext, rule = c_Ext)
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

def extend_ch(ch_df, params = {'N': 2, 'D13':2, 'Qt': 2, 'Pt':0.1}):
    
    lm = LinearRegression()
    lm.fit(ch_df.iloc[:,:-1],ch_df.iloc[:,-1])
    ext_df = ch_df.iloc[:,:-1].copy()
    plus_df = ch_df.iloc[:,:-1].copy()
    mins_df = ch_df.iloc[:,:-1].copy()
    for col in ext_df.columns:
        plus_df[col] = plus_df[col] + params[col]
        mins_df[col] = mins_df[col] - params[col]

    ext_df = ext_df.append([plus_df,mins_df])

    ch = ConvexHull(ext_df)
    ext_df = ext_df.iloc[ch.vertices]
    ext_df['D0'] = lm.predict(ext_df)
    
    return ext_df
    
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