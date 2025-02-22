import numpy as np
import pandas as pd
from pyomo.environ import *
from df2block import *
from pyomo_utils import list_constraint,list_ovjective,gen_report
from pyomo.environ import *
from sympy import symbols, sympify, parse_expr

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
        Value=Value.replace('\n','').replace(',\n',',').replace(' ','').replace(', \n',',').replace("'",'')
        return Value.split(',')
    else:
        #print(type(Value))
        return Value
        
def add_Lims_2Model(m,Lims):
    # Добавление ограничений на переменные
    # m оптимизационная модель
    # Lims ограничения
    DV=dict_vars(m,Stages=True,max_point=3)
    for i in range(Lims.shape[0]):
        Temp=Lims.iloc[i]
        # Добавляем Max
        VarName=Temp.Obj+'.'+Temp.Var
        if VarName in DV.keys():
                if not (Temp.Max is np.NAN): 
                     DV[VarName].setub(Temp.Max)
                if not (Temp.Min is np.NAN): 
                    DV[VarName].setlb(Temp.Min)
                print(DV[VarName].name,'lb:',Temp.Min,' ub:',Temp.Max)
        else: 
            print ('!!!!! Ошибка в имени переменной',VarName,'!!!!')  

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
            if 'no_bounds_Vars' in varargs.keys():
                B=Block_Off_State(t,Vars=varargs['addvars'],Free_Vars=varargs['no_bounds_Vars'])   
            else:
                B=Block_Off_State(t,Vars=varargs['addvars'])   
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
            if 'MF' not in list_vars(m):
                m.MF=Var(bounds=(-10000,10000))
            
            if 'cMF[1]' not in list_constraint(m):
                m.cMF=ConstraintList()
                m.cMF.add(m.MF==0)
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
        d_v=dict_vars(m)
        for temp in d_v.keys():
            # Преобразуем имя оптимизационной переменной в имя показателя
            v=d_v[temp]
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

def get_Vars_N_D0(obj):
    DV=dict_vars(obj,Stages=True,max_point=3)
    # Ищем вхождения N и D0
    d={}
    lengthN=100
    lengthD0=100
    for VarName in DV:
        if 'N' in VarName and lengthN>len(VarName):
            lengthN=len(VarName)
            d['N']=DV[VarName]
        if 'D0' in VarName and lengthD0>len(VarName):
            lengthD0=len(VarName)
            d['D0']=DV[VarName]
    return d        
        
def set_MF(m,Type,accuracy_dh):  # Более корректная интерпретация
    # Набросок определения целевой функции
    # m - модель
    # Type - тип целевой функции
    # accuracy_dh - словарь с точностями измерителей и значениями приборов учёта

    m_=m.clone()
    # Определяем переменные с именами D0 и N
    d=get_Vars_N_D0(m_)
    
    t=0
    if 'cMF[1]' not in list_constraint(m_):
        m_.cMF=ConstraintList()
        m_.cMF.add(m_.MF==0)
        m_.cMF[1].deactivate()
    else:
        for i in m_.cMF.keys():
            m_.cMF[i].deactivate()

    if 'SEAC' in Type:
        add_stat_data(m_,accuracy_dh) 
        m_.O = Objective(expr= m_.MF, sense=minimize)
    elif 'Dmin' in Type:
        m_.cMF.add(m_.MF==d['D0'])#m_.Vars[t,'D0'])
        m_.O = Objective(expr= m_.MF, sense=minimize)
    elif 'Dmax' in Type:
        m_.cMF.add(m_.MF==d['D0'])#-m_.Vars[t,'D0'])
        m_.O = Objective(expr= m_.MF, sense=maximize)
    elif 'Nmin' in Type:  
        m_.cMF.add(m_.MF==d['N'])#m_.Vars[t,'N'])    
        m_.O = Objective(expr= m_.MF, sense=minimize)
    elif 'Nmax' in Type:
        m_.cMF.add(m_.MF==d['N'])#-m_.Vars[t,'N'])
        m_.O = Objective(expr= m_.MF, sense=maximize)
    return m_    

def get_Blocks(Blocks,name):
    def get_block_(b,name):
        return Blocks[name].clone()
    return Block(name,rule=get_block_)        
   
def fix_vars(obj,DF,hi=0):
    # Фиксируем переменные
    dictVars=dict_vars(obj,Stages=True,max_point=3)
    keysDictVars=dictVars.keys()
    keysDF=DF.keys()
    if isinstance(hi,int):
        iDF=DF.iloc[hi]
    else:
        iDF=DF.loc[hi]
    for DFVar in keysDF:
        if DFVar in keysDictVars:
            value=iDF[DFVar]
            print(f'fix {DFVar} :{value}')
            dictVars[DFVar].fix(value)
            
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
    
def correct_varname(name):
    # Скорректировать имена
    name=name.replace('BoolVars[','')
    name=name.replace('Vars[0,','').replace(']','')
    name=name.replace('[','')
    name=name.replace('Turbines','')
    name=name.replace('Obj','')
    name=name.replace('Turbine','')
    name=name.replace('Boilers','')
    name=name.replace('REU','')
    name=name.replace('PVDPVD','PVD')
    name=name.replace('Stages','')
    name=name.replace("'",'')
    return name    
    
def dict_vars(m,Stages=False,max_point=2):
    Vars={}
    if 'Stages' in m.name:
        max_point=max_point+1
        Stages=True
    for Var1 in m.component_data_objects(Var):
        name=Var1.name
        if ('Stages'not in name) or Stages:
            if name.count('.')<max_point:
                name=correct_varname(name)
                Vars[name]=Var1
    return Vars 
    
def vars_to_dataframe(m,Stages=False,max_point=2):
    Vars={}
    if 'Stages' in m.name:
        max_point=max_point+1
        Stages=True
    for Var1 in m.component_data_objects(Var):
        name=Var1.name
        if ('Stages'not in name) or Stages:
            if name.count('.')<max_point:
                name=correct_varname(name)
                Vars[name]=[Var1.value]
    return pd.DataFrame(Vars)

def Block2Model(m_):
    # Конвертация фрагмента модели в модель
    # Block2Model
    name=m_.name
    name=correct_varname(name)
    Blocks_={name:m_}
    obj=ConcreteModel()
    obj.Obj=get_Blocks(Blocks_,[name])
    obj.t=[0]
    obj.MF=Var()
    return obj

def add_Eq_In_STR(TBlock,eq,DV=None):
    BNAME=get_block_name(TBlock.name)
    if DV==None:
        DV={}
    temp=dict_vars(TBlock,max_point=3) # Допускаем, что локальные переменные могут содержать 1 '.'
    if len(BNAME)>0:
        temp=dict((key.replace('Stages','').replace(BNAME+'.',''), value) for (key, value) in temp.items())        
    DV={**DV,**temp}
    # Переименование DV в случае, если локально работаем
    
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
    
def add_Equestions(TBlock,eqs,DV=None):  
    LCs=list_constraint(TBlock)
    Status=[]
    if ('CL[1]' not in LCs) and (TBlock.name+'.'+'CL[1]' not in LCs):
        TBlock.CL=ConstraintList()
    for eq in eqs:
        try:
            Vars_Not_Found=add_Eq_In_STR(TBlock,eq,DV=DV) 
            if len (Vars_Not_Found)==0:
                Status.append(pd.DataFrame({'Obj':[TBlock.name],'Eq':[eq],'Status':['OK']}))
            else:
                Status.append(pd.DataFrame({'Obj':[TBlock.name],'Eq':[eq],'Status':['No Vars:'+';'.join(Vars_Not_Found)]}))
        except:
            print('Error in:', eq)
            Status.append(pd.DataFrame({'Obj':[TBlock.name],'Eq':[eq],'Status':['Error']}))
    return  pd.concat(Status)
            

def add_Block_Stages_to_DF(DF_Eq):
    # Добавление колонки Block_Stages в DataFrame
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
    
def her(m):
    return True   

def add_Equestions_To_Objs(m,DF_Eq):
    DE=Dict_EQ(DF_Eq)
    print('Добавляемые уравнения в объекты:')
    print(DE)
    Status=[]
    BlockDict=dict_blocks(m,max_count_point=1)
    DV=dict_vars(m)
    for k in DE.keys():
        print('-----!!!!!!!!!!!!!!!   ',k,'        !!!!!!------')
        Eq=DE[k]
        TBlock=BlockDict[k]
        Status.append(add_Equestions(TBlock,Eq,DV=DV))
    Status=pd.concat(Status)    
    return Status 
