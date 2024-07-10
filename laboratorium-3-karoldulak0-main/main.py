import numpy as np
import scipy
import pickle
import math
from typing import Union, List, Tuple


def absolut_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu bezwzględnego. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu bezwzględnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    #sprawdzam czy mają dobre typy 
    if isinstance(v , (int,float,List,np.ndarray)) and isinstance(v_aprox,(int,float,List,np.ndarray)):
        
        #liczba liczba
        if isinstance(v,(int,float)) and isinstance(v_aprox,(int,float)):
            return np.abs(v - v_aprox)

        # lista - lista 
        elif isinstance(v,List) and isinstance(v_aprox, List):
            if len(v) == len(v_aprox):
                return np.abs(np.array(v) - np.array(v_aprox))
            else:
                return np.NaN
        
        #wektor wektor 
        elif isinstance(v, np.ndarray) and isinstance(v_aprox , np.ndarray):
            zipped = zip(v.shape[::-1] , v_aprox.shape[::-1])
            if all((m == n) or (n == 1) or (m == 1) for m,n in zipped ):
                return np.abs(v - v_aprox)
            else:
                return np.NaN
         

        #liczba - lista 
        elif isinstance(v,(int,float)) and isinstance(v_aprox, List):
            res = np.zeros(len(v_aprox), dtype = int )
            for i in range(len(v_aprox)):
                res[i] = np.abs(v - v_aprox[i])
            return res 

        #lista - liczba
        elif isinstance(v , List) and isinstance(v_aprox, (int, float)):
            res = np.zeros(len(v), dtype = int )
            for i in range(len(v)):
                res[i] = np.abs(v[i] - v_aprox)
            return res 

        #wektor - liczba , liczba  - wektor 
        elif isinstance(v, np.ndarray) and isinstance(v_aprox, (int, float)) or isinstance(v_aprox, np.ndarray) and isinstance(v, (int, float)):
            return np.abs(v - v_aprox)  

        #lista - wektor 
        elif isinstance(v, List) and isinstance(v_aprox, np.ndarray):
            if len(v) == v_aprox.shape[0]:
                res = np.zeros(len(v), dtype = int )
                for i in range(len(v)):
                    res[i] = np.abs(v[i] - v_aprox[i])
                return res
            else:
                return np.NaN

        #wektor lista
        elif isinstance(v, np.ndarray) and isinstance(v_aprox, List):
            if len(v_aprox) == v.shape[0]:
                res = np.zeros(len(v_aprox), dtype=int)
                for i in range(len(v_aprox)):
                    res[i] = np.abs(v[i] - v_aprox[i])
                    return res
            else:
                return np.NaN
    else:
        return np.NaN


def relative_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu względnego.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu względnego,
                                       NaN w przypadku błędnych danych wejściowych
    """

    #blad wzgledny to blad bezwzgledny tylko podzielony jeszcze przez wartosc dokladna takze uzyje tamtej funkcji przy wyznaczaniu tego bledu:

    abs_err = absolut_error(v , v_aprox)

    #jezeli funkcje zroci NaN 
    if abs_err is np.NaN:
        return np.NaN 
    #gdy v = 0 
    elif isinstance(v, (int ,float )) and v == 0:
        return np.NaN
    #gdy v jest wektorem 
    elif isinstance(v , np.ndarray):
        return np.divide(abs_err,v)
    #gdy v jest wektorem i jest pusty 
    elif isinstance(v, np.ndarray) and not v.any():
        return np.NaN
    #wektor - lista 
    elif isinstance(abs_err , np.ndarray) and isinstance(v , List):
        result = np.zeros(len(v))
        for i in range(len(v)):
            if v[i] == 0:
                return np.NaN
            result[i] = abs_err[i] / v[i]
        return result
    else:
        return abs_err / v  


def p_diff(n: int, c: float) -> float:
    """Funkcja wylicza wartości wyrażeń P1 i P2 w zależności od n i c.
    Następnie zwraca wartość bezwzględną z ich różnicy.
    Szczegóły w Zadaniu 2.
    
    Parameters:
    n Union[int]: 
    c Union[int, float]: 
    
    Returns:
    diff float: różnica P1-P2
                NaN w przypadku błędnych danych wejściowych
    """

    
    if isinstance(n, int) and isinstance(c, (int, float)):
        b = 2 ** n
        p1 = b - b + c
        p2 = b + c - b
        return np.abs(p1-p2)
    else:
        return np.NaN


def exponential(x: Union[int, float], n: int) -> float:
    """Funkcja znajdująca przybliżenie funkcji exp(x).
    Do obliczania silni można użyć funkcji scipy.math.factorial(x)
    Szczegóły w Zadaniu 3.
    
    Parameters:
    x Union[int, float]: wykładnik funkcji ekspotencjalnej 
    n Union[int]: liczba wyrazów w ciągu
    
    Returns:
    exp_aprox float: aproksymowana wartość funkcji,
                     NaN w przypadku błędnych danych wejściowych
    """

    if isinstance(x, (int, float)) and isinstance(n, int):
        
        if n < 0:
            return np.NaN
        else:
            e = 0 
            for i in range(n):
                e += (x ** i) * (1 / math.factorial(i))
            return e
    else:
        return np.NaN
    


def coskx1(k: int, x: Union[int, float]) -> float:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 1.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx float: aproksymowana wartość funkcji,
                 NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(k, (int)) and isinstance(x , (int,float)):
        if k < 0: 
            return np.NaN
        elif k == 0: 
            return 1 
        elif k == 1:
            return np.cos(x)
        elif k > 1 : 
            return 2 * np.cos(x) * coskx1(k-1 , x) - coskx1(k-2, x )
    else:
        return np.NaN    


    


def coskx2(k: int, x: Union[int, float]) -> Tuple[float, float]:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 2.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx, sinkx float: aproksymowana wartość funkcji,
                        NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(k, (int)) and isinstance(x , (int,float)):
        if k < 0: 
            return np.NaN
        elif k == 0: 
            return 1,0 
        elif k == 1:
            return np.cos(x) , np.sin(x)
        elif k > 1 : 
            return np.cos(x)*coskx2(k-1,x)[0] - np.sin(x) * coskx2(k-1,x)[1] , np.sin(x) * coskx2(k - 1 , x)[0] + np.cos(x) * coskx2(k - 1, x)[1]
    else:
        return np.NaN       


    



