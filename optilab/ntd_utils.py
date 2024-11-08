# Работа с характеристиками оборудования

# Метод для аппроксимации функций
import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator # импортируем методы интерполяции
def add_curve(Curves,Name,X,F):
        n=np.shape(X);
        if len(n)==1: # Интерполяция одномерных функций
            Curves.update({Name:interp1d(X,F,bounds_error=False, fill_value='extrapolate')})
        else:         # Интерполяция многомерных функций
            Curves.update({Name:LinearNDInterpolator(X, F,rescale=True)})
        return Curves
