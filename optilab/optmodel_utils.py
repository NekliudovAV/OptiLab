import numpy as np
from pyomo.environ import *
from df2block import *
from pyomo_utils import list_constraint,list_ovjective,gen_report
from clsTurbineOpt import select_indexes
from pyomo.environ import *
from sympy import symbols, sympify

# Для стандартизации расчётной модели удобнее вынести управление генерации оптимизационной модели в таблшичный файл
# Представленный ниже код помогнает реализовать эту идею

def select_DF(Data,DF_keys):
    # Получение DataFrame из "Иерархического" словаря по ключу DF_keys
    if (DF_keys is None) or (DF_keys is np.NAN):        
        return None
    DF_keys=DF_keys.split('.')
    Temp=Data
   
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

def create_Blocks(Data,DF_Objects):
    # Data словарь с харакетристиками
    # DF_Objects правила формирования блоков модели
    # Формирование блоков для модели
    t=range(1)
    Blocks={}
    for i in range(DF_Objects.shape[0]):
        TD=DF_Objects.iloc[i]
        varargs={}
        for tk in TD.keys():
            if tk in ['DF']:
                DF=select_DF(Data,TD.DF)
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
                accuracy_dh.update({ key : {'s':1,'meas':value}})
            elif 'Dd' in key:
                accuracy_dh.update({ key : {'s':10,'meas':value}})
            elif 'Thv' in key:    
                accuracy_dh.update({ key : {'s':2,'meas':value}})    
            elif 'Qt' in key:    
                accuracy_dh.update({ key : {'s':5,'meas':value}})
            elif 'D0' in key:    
                accuracy_dh.update({ key : {'s':10,'meas':value}})
            elif 'QSUV' in key:    
                accuracy_dh.update({ key : {'s':10,'meas':value}})
            elif 'GSUV' in key:    
                accuracy_dh.update({ key : {'s':50,'meas':value}})    
            elif 'TrPSG' in key:
                accuracy_dh.update({ key : {'s':1,'meas':value}}) 
            elif 'D2_5' in key:
                accuracy_dh.update({ key : {'s':5,'meas':value}})   
            elif 'Gd' in key:
                accuracy_dh.update({ key : {'s':100,'meas':value}})   
            elif 'IWF'in key:  
                accuracy_dh.update({ key : {'s':0.3,'meas':value}})
            elif 'IFW'in key:  
                accuracy_dh.update({ key : {'s':0.3,'meas':value}})
    
            else:
                accuracy_dh.update({ key : {'s':100,'meas':value}})
                
        sigma={'N_130':0.2,'TA3_Qsuv':1,'N_30':0.2,'DsTOKЕM23_8':1,
               'QPU1':.5,'QPU2':.5,'QPU3':.5,'QNet1':.5,'QNet2':.5,'QNet3':.5,'QNet4':.5,
               'D_B1':50,'D_B2':50,'D_B3':50,'D_B4':50,
               'D_PWD3_8':10,'De_T':5, 'De_130':5,'De_30':5,'RU18_D0':10,'RU16_D0':10,'RU2_D0':20,'Dpsuv':5,'Dd':10,'QSUV':15,'TA5.D0':15,'TA5.P0GPZ':1}
        
        meas={}
        # Пример задания сигмы
        for sn in sigma.keys():
            if sn in accuracy_dh.keys():
                accuracy_dh[sn]['s']=sigma[sn]
        for mn in meas.keys():
            if sn in accuracy_dh.keys():
                accuracy_dh[mn]['meas']=meas[mn]
                
        for key in accuracy_dh.keys():
            if accuracy_dh[key]['meas']<1:
                accuracy_dh[key]['s']=1
                accuracy_dh[key]['meas']=0
              
        print(accuracy_dh)
        return accuracy_dh                 
        
def find_constr(T11,name_constr='res_list'):
    # Проверка: есть ли ограничение с заданным именем в модели
    temp_=list_constraint(T11)
    exist=False
    for i in temp_:
        if name_constr in i:
            exist=True
    return  exist  

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
            if 'MF' not in list_vars(m):
                m.MF=Var(bounds=(-10000,10000))
            
            if 'cMF[1]' not in list_constraint(m):
                m.cMF=ConstraintList()
                m.cMF.add(m_.MF==0)
                m.cMF[1].deactivate()

            else:
                for i in m.cMF.keys():
                    m.cMF[i].deactivate()
            

            m.cMF.add(m.MF==m.Imbalance)
            
            # Если целевая функция не определена, то:
            if 'O' not in list_ovjective(m):
                print('-----------set m.O-------------')
                m.O = Objective(expr= m.MF, sense=minimize)
                print(list_ovjective(m))


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
                print(temp)
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
        
def get_block(b,name):
    return Blocks[name].clone()      

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

def dict_vars(m):
    Vars={}
    for Var1 in m.component_data_objects(Var):
        name=Var1.name
        if name.count('.')<2:
            name=name.replace('BoolVars[','')
            name=name.replace('Vars[0,','').replace(']','')
            
            Vars[name]=Var1
    return Vars 

def add_Eq_In_STR(TBlock,eq):
    DV=dict_vars(TBlock)
    if '<=' in eq:
        Type ='InEq'
        eq=eq.split('<=')
    elif '=' in eq:
        Type = 'Eq' 
        eq=eq.split('=')
    string = eq[1]+'-'+eq[0]
    expr = sympify(string)
    args=list(expr.args)

    print(expr)
    free_syms = sorted(expr.free_symbols, key = lambda symbol: symbol.name)
    if args[0].is_constant():
        konstant=float(args[0])
    else:
        konstant=0

    expr_=konstant
    for fs in free_syms: 
        print(fs,'Коэффициент: ',expr.coeff(fs))
        expr_=expr_+float(expr.coeff(fs))*DV[str(fs)]
    # Определение константы:
    args=list(expr.args)
    print('Константа:',konstant)
    if Type in ['InEq']:
        TBlock.CLE.add(expr=expr_<=0)
    elif Type in ['Eq']:    
        TBlock.CLE.add(expr=expr_==0)
    #TBlock.CL.pprint()    
    
def add_Equestions(TBlock,eqs):  
    if 'CLE[1]' not in list_constraint(TBlock):
        TBlock.CLE=ConstraintList()
    for eq in eqs:
        add_Eq_In_STR(TBlock,eq)    
