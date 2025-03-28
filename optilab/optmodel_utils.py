import numpy as np
from tqdm import tqdm_notebook as tqdm
from joblib import Parallel, delayed, cpu_count
from pyomo.environ import *
from df2block import *
from pyomo_utils import list_constraint,list_ovjective,gen_report,gen_report2
from pyomo.environ import *
from sympy import symbols, sympify, parse_expr

# Для стандартизации расчётной модели удобнее вынести управление генерации оптимизационной модели в таблшичный файл
# Представленный ниже код помогнает реализовать эту идею

def select_DF(Curvs,DF_keys):
    # Получение DataFrame из "Иерархического" словаря по ключу DF_keys
    if (DF_keys is None) or (DF_keys is np.NAN):        
        return None
        
    if DF_keys in Curvs.keys():
        return Curvs[DF_keys]
    
    DF_keys=DF_keys.split('.')
    Temp=Curvs
   
    for i in range(len(DF_keys)):
        if DF_keys[i] in Temp.keys():
            Temp=Temp[DF_keys[i]]
        else:
            print('Не найден датафрейм:', DF_keys, ' есть ', Temp.keys())
            return None
    DF_=Temp
    return DF_
    
   

def Prepare_Var(Value):
    # Преобразование Str в список. Запятые являются разделителями
    if (Value is None) or (Value is np.NAN):        
        return None
    elif isinstance(Value,str):        
        return Value.split(',')
    else:
        #print(type(Value))
        return Value
        
def add_Lims_2Model(m,Lims):
    # Добавление ограничений на переменные
    # m оптимизационная модель
    # Lims ограничения
    for i in range(Lims.shape[0]):
        Temp=Lims.iloc[i]
        # Добавляем Max
        if Temp.Var in m.Turbine[Temp.Obj].VarNames:
            if not (Temp.Max is np.NAN): 
                m.Turbine[Temp.Obj].Vars[0,Temp.Var].setub(Temp.Max)
            if not (Temp.Min is np.NAN): 
                m.Turbine[Temp.Obj].Vars[0,Temp.Var].setlb(Temp.Min)
            print(m.Turbine[Temp.Obj].Vars[0,Temp.Var].name,'lb:',Temp.Min,' ub:',Temp.Max)
        else: 
                print ('Ошибка в имени переменной',Temp.Var,'! Определены:', m.Turbine[Temp.Obj].VarNames)     

def create_Blocks(Curvs,DF_Objects):
    # Curvs словарь с харакетристиками
    # DF_Objects правила формирования блоков модели
    # Формирование блоков для модели
    t=range(1)
    Blocks={}
    for i in range(DF_Objects.shape[0]):
        TD=DF_Objects.iloc[i]
        print('-------',TD['Name'],'-------')
        varargs={}
        for tk in TD.keys():
            
            if tk in ['DF']:
                DF=select_DF(Curvs,TD.DF)
            elif tk in['func']:    
                func=TD['func']
            elif tk in['Name']:
                Name=TD['Name']
            elif tk in ['block']:    
                value=Prepare_Var(TD[tk])
                if not value is None:
                    block=[] # Формирутся список блоков
                    for nameBlock in value:
                        block.append(Blocks[nameBlock].clone())
                    varargs[tk]=block
            else:
                value=Prepare_Var(TD[tk])
                print(tk,':',value)
                if not value is None:
                    varargs[tk]=value

        print('-------------Name:-----------------',Name)
        print('varargs: ',varargs)

        if func in ['CH_Block']:
            if not DF is None:
                B=CH_Block(t, DF, **varargs)
            else:
                print('!!!!________________Error in Name DF__________!!!!!!') 
                Error
        elif func in ['PWL_Block']:    
            if not DF is None:
                B=PWL_Block(t, DF, **varargs)
            else:
                print('!!!!________________Error in Name DF__________!!!!!!') 
                Error
        elif func in ['N_Stages']:     
            # Удаление 
            varargs.pop('Type')
            varargs.pop('Obj')
            varargs.pop('block')
            B=N_Stages(t,*block,**varargs)
        elif func in ['Block_Off_State']:         
            B=Block_Off_State(t,Vars=varargs['addvars'],Free_Vars=varargs['no_bounds_Vars'])   
        Blocks[Name]=B
    return Blocks
    
def get_sigma_U(df,i=0):
        # df - Датафрейм с данными
        # i - номер строки DataFrame
        # Опредление сигмы по имени показателя. Не унифицировано, набросок.
        accuracy_dh={}
        for key in df.keys():
            if type(i) is type(df.index[0]):
                value=df[key].loc[i]
            else:
                value=df[key].iloc[i]
                
            if 'N' in key:
                s=1
            elif 'Dd' in key:
                s=10
            elif 'P0' in key:
                s=1
            elif 'P2' in key:
                s=0.1
                
            elif 'Thv' in key:    
                s=2
            elif 'Qt' in key:    
                s=5
            elif 'D0' in key:    
                s=10
            elif 'QSUV' in key:    
                s=10
            elif 'GSUV' in key:    
                s=50
            elif 'TrPSG' in key:
                s=1
            elif 'D2_5' in key:
                s=5
            elif 'Gd' in key:
                s=100
            elif 'IWF'in key:  
                s=0.3
            elif 'T' in key:
                s=1
            else:
                s=100
            accuracy_dh.update({key: {'s':s,'meas':value}})
            
        # Можем симы задавать "вручную"
        manual_sigma(accuracy_dh)
        #print(accuracy_dh)
        return accuracy_dh     
        
def manual_sigma(accuracy_dh,sigma=None):
        if sigma==None:
            sigma={'N_130':0.2,'TA3_Qsuv':1,'N_30':0.2,'DsTOKЕM23_8':1,
               'QPU1':.5,'QPU2':.5,'QPU3':.5,'QNet1':.5,'QNet2':.5,'QNet3':.5,'QNet4':.5,
               'D_B1':50,'D_B2':50,'D_B3':50,'D_B4':50,
               'D_PWD3_8':10,'De_T':5, 'De_130':5,'De_30':5,'RU18_D0':10,'RU16_D0':10,'RU2_D0':20,'Dpsuv':5,'Dd':10,'QSUV':15,'TA5.D0':15,'TA5.P0GPZ':1}
        # Пример задания сигмы
        for sn in sigma.keys():
            if sn in accuracy_dh.keys():
                accuracy_dh[sn]['s']=sigma[sn]           
        
def find_constr(T11,name_constr='res_list'):
    # Проверка: есть ли ограничение с заданным именем в модели
    temp_=list_constraint(T11)
    exist=False
    for i in temp_:
        if name_constr in i:
            exist=True
    return  exist  
    
def select_indexes(g_range2):
    k=0
    ind=[0]
    last=g_range2[0]
    g_range2
    for i in range(1,len(g_range2)):
        last=g_range2[k-1]
        now_=g_range2[k]
        k+=1
        next_=g_range2[k]
        if (last!=now_)|(next_!=now_):
            ind.append(k-1)
    # добавляем последний
    ind.append(k)
    return np.unique(ind)     

def add_stat_data(m,accuracy_dh,N=64,max_s2=100,min_s2=0,bound_s=10):
        # Добавление блока небалансовых уравнений
        # me - список переменых, по которым присутствует статистика
        # accuracy_dh [key]['s']    - сигма отклонений
        # accuracy_dh [key]['meas'] - значение по прибору учёта (меняется каждый час)

        #N = 64
        #max_s2=9
        #min_s2=0
        #bound_s=10

        neg_dW = -bound_s
        pos_dW = bound_s

        g_step = (pos_dW-neg_dW)/N
        g_range = np.arange(neg_dW, pos_dW+1e-3, g_step)
        g_range2 =(g_range**2).clip(min_s2,max_s2)
        s_indexes=select_indexes(g_range2)
        g_range2=g_range2[s_indexes]
        g_range=g_range[s_indexes]

        qdf = pd.DataFrame({'dW': g_range, 'dW2': g_range**2})

        me=list(accuracy_dh.keys())
        if find_constr(m,name_constr='res_list'):
            # При последующих вызоывах
            for i in m.res_list.index_set():
                m.res_list[i].deactivate()
        else:
            # При первом вызове формируются конструкции
            m.residual = Var(m.t, me, bounds=(-9.99,9.99))
            m.res2 = Var(m.t, me)
            m.res_list = ConstraintList()


            def v2(b,t,var):
                b.PWL = Piecewise(m.res2[t,var], m.residual[t,var], pw_pts=list(qdf['dW'].values),
                                  pw_constr_type='EQ', f_rule=list(qdf['dW2'].values), pw_repn='SOS2')

            m.quad_v = Block(m.t, me, rule=v2)

            m.Imbalance = Var()
            m.imbalance = Constraint(expr= m.Imbalance == sum(m.res2[t,meas] for meas in me for t in m.t))
            # Определение целевой функции
            MFmax=10000
            if 'MF' not in list_vars(m):
                m.MF=Var(bounds=(-MFmax,MFmax))
            
            if 'cMF[1]' not in list_constraint(m):
                m.cMF=ConstraintList()
                m.cMF.add(m.MF<=MFmax)
                m.cMF[1].deactivate()

            else:
                for i in m.cMF.keys():
                    m.cMF[i].deactivate()
            

            m.cMF.add(m.MF==m.Imbalance)
            
            # Если целевая функция не определена, то:
            if 'O' not in list_ovjective(m):
                #print('-----------set m.O-------------')
                m.O = Objective(expr= m.MF, sense=minimize)
                #print(list_ovjective(m))


        # Добавляем ограничения      
        for v in m.component_data_objects(Var):
            # Преобразуем имя оптимизационной переменной в имя показателя
            
            # Было: Turbine[T11].Vars[0,D0]
            temp=v.name.replace('Vars[0,','').replace(']','').replace('[','') #Убираем Vars[0, ]
            # Остаётся TurbineT11.D0
            Types=['Turbine','Boiler']
            for T in Types:
                 temp=temp.replace(T,'') # Убиарем  Turbine
                 #temp=temp.replace('Turbine','') # Убиарем  Turbine[        
            # Остаётся T11.D0
            
            
            if temp in me:
                # При найденых значениях вводим параметры с отклонениями
                #print(temp)
                for t in m.t:
                    #try:
                        m.res_list.add(m.residual[t,temp] == (v - accuracy_dh[temp]['meas'])/accuracy_dh[temp]['s'])
                    #except:
                        if accuracy_dh[temp]['s']==0:
                            print('Деление на 0: сигма = 0 ', temp )
                            0/0
                        elif np.isnan(accuracy_dh[temp]['meas']):
                            print('Значение не определено! ', temp)
                            0/0


        return m        
        
def set_MF(m_,Type,accuracy_dh):  # Более корректная интерпретация
    # Набросок определения целевой функции
    # m_ - модель
    # Type - тип целевой функции
    # accuracy_dh - словарь с точностями измерителей и значениями приборов учёта
    t=0
    if 'cMF[1]' not in list_constraint(m_):
        m_.cMF=ConstraintList()
        m_.cMF.add(m_.MF==0)
        m_.cMF[1].deactivate()
    else:
        for i in m_.cMF.keys():
            m_.cMF[i].deactivate()
    # Если 
    if 'SEAC' in Type:
        add_stat_data(m_,accuracy_dh) 
    elif 'Dmin' in Type:
        m_.cMF.add(m_.MF==m_.Vars[t,'D0'])#+
                #((m.Neb[t,'D30_plus']+m.Neb[t,'D30_minus'])*1+
                #(m.Neb[t,'D13_plus']+m.Neb[t,'D13_minus'])*1+
                #(m.Neb[t,'D7l_plus']+m.Neb[t,'D7l_minus'])*0.7+
                #(m.Neb[t,'D7r_plus']+m.Neb[t,'D7r_minus'])*0.7+
                #(m.Neb[t,'D1_2_plus']+m.Neb[t,'D1_2_minus'])*0.5)*10)
    elif 'Nmin' in Type:  
        m_.cMF.add(m_.MF==m_.Vars[t,'D0'])    
    elif 'Nmax' in Type:
        m_.cMF.add(m_.MF==-m_.Vars[t,'D0'])
    
    if 'O' not in list_ovjective(m_):
        print('-----------set m.O-------------')
        print(list_ovjective(m_))
        m_.O = Objective(expr= m_.MF, sense=minimize)

def get_Blocks(Blocks,name):
    def get_block_(b,name):
        return Blocks[name].clone()
    return Block(name,rule=get_block_)        
   

# Выполнение расчёта для интервала времени
def calculate_(m,FData,calctype='Dmin'):
    # Старт расчёта
    temp=[]
    m1=m.clone()
    for i in range(1):#FData.shape[0]):
        # Формируем целевую 
        accuracy_dh=get_sigma_U(FData,i)
        #calctype='SEAC'
        #calctype='Dmin'
        #calctype='Dmax'
        set_MF(m1,calctype,accuracy_dh)
        # Расчёт 
        opt = SolverFactory('scip')
        opt.options["limits/time"] = 20
        status = opt.solve(m1, tee=True)
        # Формирование отчёта
        res=gen_report(m1).iloc[0:1]
        temp.append(res[[i for i in res.keys() if i.count('.')<2]])
    res_out=pd.concat(temp)
    return res_out    

    
def calculate_SE(model,SE,i=0):
    # m
    try:
        if callable(model):
            m=model()
        else:
            m=model
        accuracy_dh=get_sigma_U(SE,i=i)
        add_stat_data(m,accuracy_dh)
        opt = SolverFactory('scip')
        opt.options["limits/gap"] = 0.01
        opt.options["limits/time"] = 20
        opt.options["numerics/feastol"]=0.001
        # Пишем словарь с соотвествием переменных в оптимизационной модели и статистики
        opt.solve(m, tee=False,symbolic_solver_labels=True)
        res=gen_report2(m)
        res.index=[SE.index[i]]
    except:
        res=[]
    return res

# Мультипоточный расчёт

def calc_multicore_SE(model,DF_SE):      
        # calculate_SE - ссылка на функцию, в которой выполняется оценка состояния
        # Аргумент 1: 
        func=calculate_SE
        n=DF_SE.shape[0]
        k=100
        results=[]
        
        for i in range(0,int(np.ceil(n/k)),1):
                    temp=range(i*k,min((i+1)*k,n),1)
                    results.extend(Parallel(n_jobs=10, verbose=6)(delayed(func)(model,DF_SE,fl) for fl in temp))#range(0,n,1)))
                    print('processed: %d' %(i*k))
        # Выбираем только строчки, по которым расчёты были выполнены.
        results=pd.concat([i for i in  results if len(i)>0])            
        return results    
    
    
def dict_vars(m,Stages=False):
    Vars={}
    max_point=2
    if 'Stages' in m.name:
        max_point=3
        Stages=True
    for Var1 in m.component_data_objects(Var):
        name=Var1.name
        if ('Stages'not in name) or Stages:
            if name.count('.')<max_point:
                name=name.replace('BoolVars[','')
                name=name.replace('Vars[0,','').replace(']','')
                name=name.replace('[','')
                name=name.replace('Turbines','')
                name=name.replace('Turbine','')
                name=name.replace('Boilers','')
                name=name.replace('REU','')
                Vars[name]=Var1
    return Vars 

def add_Eq_In_STR(TBlock,eq):
    DV=dict_vars(TBlock)
    # Переименование DV в случае, если локально работаем
    BNAME=get_block_name(TBlock.name)
    if len(BNAME)>0:
        DV=dict((key.replace('Stages','').replace(BNAME+'.',''), value) for (key, value) in DV.items()) 
    
    if '<=' in eq:
        Type ='InEq1'
        eq=eq.split('<=')
    elif '>=' in eq:
        Type ='InEq2'
        eq=eq.split('>=')
    elif '=' in eq:
        Type = 'Eq' 
        eq=eq.split('=')
    string = eq[1]+'-'+eq[0]
    #expr = sympify(string)
    #string=string.replace('.','_point_')
    string=re.sub(r'\w+\.[^\d]', lambda x: x.group(0).replace('.', '_point_'), string)
    expr=parse_expr(string,local_dict={'N':symbols('N')})
    args=list(expr.args)

    print(expr)
    free_syms = sorted(expr.free_symbols, key = lambda symbol: symbol.name)
    if args[0].is_constant():
        konstant=float(args[0])
    else:
        konstant=0

    expr_=konstant
    Vars_not_Found=[]
    for fs in free_syms: 
        str_fs=str(fs).replace('_point_','.')
        print(str_fs,'Коэффициент: ',expr.coeff(fs))
        if str_fs not in DV.keys():
            Vars_not_Found.append(str_fs)
            print(str_fs, 'отсутствует в списке переменных')
            print('список переменных:', list(DV.keys()))
    if len(Vars_not_Found)>0:
        return Vars_not_Found
            
    for fs in free_syms: 
        str_fs=str(fs).replace('_point_','.')
        expr_=expr_+float(expr.coeff(fs))*DV[str_fs]
        
    # Определение константы:
    args=list(expr.args)
    print('Константа:',konstant)
    if Type in ['InEq1']:
        TBlock.CL.add(expr=expr_<=0)
    elif Type in ['InEq2']:
        TBlock.CL.add(expr=expr_>=0)
    elif Type in ['Eq']:    
        TBlock.CL.add(expr=expr_==0)
    return Vars_not_Found
    
def add_Equestions(TBlock,eqs):  
    LCs=list_constraint(TBlock)
    Status=[]
    if ('CL[1]' not in LCs) and (TBlock.name+'.'+'CL[1]' not in LCs):
        TBlock.CL=ConstraintList()
    for eq in eqs:
        try:
            Vars_Not_Found=add_Eq_In_STR(TBlock,eq) 
            if len (Vars_Not_Found)==0:
                Status.append(pd.DataFrame({'Obj':[TBlock.name],'Eq':[eq],'Status':['OK']}))
            else:
                Status.append(pd.DataFrame({'Obj':[TBlock.name],'Eq':[eq],'Status':['No Vars:'+';'.join(Vars_Not_Found)]}))
        except:
            print('Error in:', eq)
            Status.append(pd.DataFrame({'Obj':[TBlock.name],'Eq':[eq],'Status':['Error']}))
    return  pd.concat(Status)
            

def add_ST_to_DF(DF_Eq):
    # Добавление колонки Block_Stages
    ST=[]
    for i in range(DF_Eq.shape[0]):
        if pd.isna(DF_Eq['Stages index'].iloc[i]):
            St=''
        else:
            St='.'+str(int(DF_Eq['Stages index'].iloc[i]))
        ST.append(DF_Eq['Block'].iloc[i]+St)
    DF_Eq['Block_Stages']=ST
       


def Dict_EQ(DF_Eq):
    # Группировка строк
    Eq_dict={}
    for i in range(DF_Eq.shape[0]):
        Eq=DF_Eq['Equestions'].iloc[i]
        if DF_Eq['Block_Stages'].iloc[i] in Eq_dict.keys():
            Eq_dict[DF_Eq['Block_Stages'].iloc[i]].append(Eq) 
        else:
            Eq_dict[DF_Eq['Block_Stages'].iloc[i]]=[Eq]
    return Eq_dict

#add_ST_to_DF(DF_Eq)    
#DE=Dict_EQ(DF_Eq)
#for k in DE.keys():
#    Eq=DE[k]
#    TBlock=BlockDict[k]
#    add_Equestions1(TBlock,Eq)
