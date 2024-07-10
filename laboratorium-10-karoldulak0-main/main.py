import numpy as np
import scipy
import pickle
import typing
import math
import types
import pickle 
from inspect import isfunction


from typing import Union, List, Tuple

def fun(x):
    return np.exp(-2*x)+x**2-1

def dfun(x):
    return -2*np.exp(-2*x) + 2*x

def ddfun(x):
    return 4*np.exp(-2*x) + 2


def bisection(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą bisekcji.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if isinstance(a , (int,float)) and isinstance(b , (int,float)) and isinstance(epsilon , float) and isinstance(iteration, int) and isfunction(f):
        if epsilon > 0 and iteration > 0:
            if f(a)*f(b) < 0:
                for i in range (iteration):
                    med = (a + b) / 2
                    if f(a) * f(med) < 0:
                        b = med
                    elif f(b) * f(med) < 0:
                        a = med
                    if abs(f(med)) < epsilon:
                        return med, i
                return (a + b) / 2, iteration
            else:
                return None 
        else:
            return None  
    else:
        return None 


def secant(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą siecznych.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if isinstance(a , (int,float)) and isinstance(b , (int,float)) and isinstance(epsilon , float) and isinstance(iteration, int) and isfunction(f):
        if epsilon > 0 and iteration > 0:
            if f(a)*f(b) < 0:
                for i in range (iteration):
                    med= b - f(b)*(b - a)/(f(b) - f(a))
                    if f(a) * f(med) < 0:
                        b = med
                    elif f(b)*f(med) < 0: 
                        a = med 
                    if abs(f(med)) < epsilon:
                        return med, i
                return b - f(b)*(b - a)/(f(b) - f(a)), iteration 
            else:
                return None 
        else:
            return None 
    else:
        return None 

def newton(f: typing.Callable[[float], float], df: typing.Callable[[float], float], ddf: typing.Callable[[float], float], a: Union[int,float], b: Union[int,float], epsilon: float, iteration: int) -> Tuple[float, int]:
    ''' Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.
    Parametry: 
    f - funkcja dla której jest poszukiwane rozwiązanie
    df - pochodna funkcji dla której jest poszukiwane rozwiązanie
    ddf - druga pochodna funkcji dla której jest poszukiwane rozwiązanie
    a - początek przedziału
    b - koniec przedziału
    epsilon - tolerancja zera maszynowego (warunek stopu)
    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if isinstance(a , (int,float)) and isinstance(b , (int,float)) and isinstance(epsilon , float) and isinstance(iteration, int):
        if f(a) * ddf(a) > 0:
            x = a
        else:
            x = b 
                
        if f(a)*f(b) < 0 and df(a)*df(b)>0 and ddf(a)*ddf(b)>0:
            for i in range(iteration):
                med = x - f(x) / df(x)
                if np.abs(med - x) < epsilon:
                    return x, i
                else:
                    x = med           
    else:
        return None 

