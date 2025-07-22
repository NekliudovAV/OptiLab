# Работа с характеристиками оборудования

import ast
import operator as op
import pandas as pd
import numpy as np
from seuif97 import *
from scipy.interpolate import interp1d, LinearNDInterpolator # импортируем методы интерполяции
from math import sqrt, sin, log 

# Расчёт вспомогательных функций
def calc_Pw(T):
        # Saturation pressure at a given temperature
        return pd.Series([tx2p(T[i],0)/ 0.0980665 for i in T.index],T.index)
    
def calc_Hw(T):
        # Enthalpy of water
        return pd.Series([tx2h(T[i],0) for i in T.index],T.index)/4.186
def calc_Hs(T):
        # Enthalpy of steam at the saturation point
        return pd.Series([tx2h(T[i],1) for i in T.index],T.index)/4.186

def calc_T(P):
        # Steam temperature at the saturation point at a given pressure
        return pd.Series([px2t((P[i]+1)* 0.0980665,1) for i in P.index],P.index)

def calc_H(P,T):
        Hs=pd.Series([pt2h((P[i]+1)* 0.0980665,T[i])  for i in T.index],T.index)
        Hs=Hs/4.186
        Hs_=pd.Series([tx2h(T[i],1) for i in T.index],T.index)
        Hs_=Hs_/4.186
        Hs[Hs_>Hs]=Hs_[Hs_>Hs]
        return Hs

def clip(df,min_,max_):
    return df.clip(min_,max_)


def add_curve(Curves,Name,X,F):
        n=np.shape(X);
        if len(n)==1: # Интерполяция одномерных функций
            Curves.update({Name:interp1d(X,F,bounds_error=False, fill_value='extrapolate')})
        else:         # Интерполяция многомерных функций
            Curves.update({Name:LinearNDInterpolator(X, F,rescale=True)})
        return Curves


class ExpressionEvaluator:
    def __init__(self, df):
        self.df = df
        self.ops = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
        }
        
        self.functions = {
            'fig':self.calc_curve,
            'clip':clip,
            'tw2p':calc_Pw,
            'tw2h':calc_Hw,
            'ts2h':calc_Hs,
            'px2t':calc_T,
            'pt2h':calc_H,
            'sqrt': np.sqrt,
            'sin': np.sin,
            'log': np.log,
            'sum': np.sum,
            # Добавьте другие функции по необходимости
        }
        self.curvs={}
    def calc_curve(self,Name,*X):
        #print('calc_curve')
        #print('Name:',Name)
        #print('X:',*X)
        return self.curvs[Name](*X)
        
    def add_curve(self,Name,X,F):
        n=np.shape(X);
        if len(n)==1: # Интерполяция одномерных функций
            self.curvs.update({Name:interp1d(X,F,bounds_error=False, fill_value='extrapolate')})
        else:         # Интерполяция многомерных функций
            self.curvs.update({Name:LinearNDInterpolator(X, F,rescale=True)})
        return self.curvs
    

    def eval_expr(self, expr):
        node = ast.parse(expr, mode='eval')
        return self._eval(node.body)

    def _eval(self, node):
        if isinstance(node, ast.Num):  # Число
            return node.n
        elif isinstance(node, ast.Str):  # Строковый литерал
            return node.s    
        elif isinstance(node, ast.Name):  # Столбец DataFrame
            return self.df[node.id]
        elif isinstance(node, ast.BinOp):  # Бинарная операция (+, -, *, /)
            left = self._eval(node.left)
            right = self._eval(node.right)
            return self.ops[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):  # Унарная операция (например, -x)
            return self.ops[type(node.op)](self._eval(node.operand))
        elif isinstance(node, ast.Call):  # Функции (sqrt(), sin() и т.д.)
            func_name = node.func.id
            args = [self._eval(arg) for arg in node.args]
            return self.functions[func_name](*args)
        else:
            raise ValueError(f"Неподдерживаемая операция: {type(node).__name__}")

    def calc_expr(self,expr,new_column_name):
        # Вычисляем выражение
        try:
            result = self.eval_expr(expr)
            print(f"Выражение:{new_column_name} = {expr} OK!\n") #Результат: {result}\n
        except Exception as e:
            print(f"Ошибка в выражении '{new_column_name}={expr}': {str(e)}")
            result = df.eval(expression)
        
        # Если результат - Series (один столбец), добавляем в DataFrame
        if isinstance(result, (pd.Series, np.ndarray)):
            self.df[new_column_name] = result
        else:
            # Если результат скалярный, применяем ко всем строкам
            self.df[new_column_name] = result
        return self.df    
        
    def calc_expressions(self,expressions):
        # Вычисление выражений
        for expr, col_name in expressions:
            print(col_name,'=',expr)
            self.calc_expr(expr, col_name)
        return  self.df 
        
    def calc_expressions_eq(self,expressions):
        # Вычисление выражений
        for expression in expressions:
            col_name, expr = expression.split('=')
            #print(col_name,'=',expr)
            self.calc_expr(expr, col_name)
        return  self.df 


def Example():
    # Функция содержит пример кострукций    
    expressions_eq=['TsKPU=Tkpu',
                'TcKPU=Tr_KPU',
                'Dsp=D2_5',
                'GsuvEG=clip(GsuvEG,0,10000)',
                'GKPU=GsuvEG',
                'Tt=(Tsob1+Tsob2)/2',       # Усреднение температуры
                
                'H0=pt2h(P0,T0)',           # Enthalpy of superheated steam
                'Hsp=pt2h(Psp,Tsp)',        # Enthalpy of industrial extraction
                'Hst=pt2h(Pt,Tt)',          # Enthalpy of heat extraction
                'HcPSG=tw2h(TcPSG)',        # Enthalpy of PSG condensate
                'HsKPU=ts2h(TcKPU)',        # Enthalpy of steam at KPU
                'dT_c=TcPSG-Tr_PSG',        # PSG temperature undercooling
                'HwKPU=tw2h(TsuvEG)',       # Enthalpy of SUV before KPU
                'Hw_KPU=tw2h(Tr_KPU)',      # Enthalpy of SUV after KPU

                'HwPSG=tw2h(TrPSG)',        # Enthalpy of water before PSG
                'Hw_PSG=tw2h(Tr_PSG)',      # Enthalpy of water after PSG

                'Tt_c=tw2p(Pt)',            # Calculation of Pt_c condensation temperature in PSG by pressure Pt
                'PcPSG=tw2p(TcPSG)',     # Saturated steam pressure at PSG condensate temperature
                'Pt_plus_1=Pt+1',           # Pressure in absolute units kgf/cm2
                'P_PSG=tw2p(Tr_PSG)',     # Saturated steam pressure at temperature

                #"D0_=fig('D0',Pt,N)"
               ]
        evaluator = ExpressionEvaluator(DF5_P)
        #evaluator.add_curve('D0',D0[['Pt','N']],D0[['D0']])
        # Добавить уравнения 
        result = evaluator.calc_expressions_eq(expressions_eq) 
        return result 
