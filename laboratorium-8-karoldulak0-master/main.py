import numpy as np
import scipy as sp
from scipy import linalg
from  datetime import datetime
import pickle

from typing import Union, List, Tuple

'''
Do celów testowych dla elementów losowych uzywaj seed = 24122022
'''

def random_matrix_by_egval(egval_vec: np.ndarray):
    """Funkcja z pierwszego zadania domowego
    Parameters:
    egval_vec : wetkor wartości własnych
    Results:
    np.ndarray: losowa macierza o zadanych wartościach własnych 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(egval_vec , (np.ndarray, List)):
        if isinstance(egval_vec[0] , str):
            return None 
        np.random.seed(24122022)
        J = np.diag(egval_vec)
        n = len(egval_vec)  
        P = np.random.rand(n,n)      
        Pinv = np.linalg.inv(P)   
        A = P@J@Pinv
        return A
    else:
        return None


def frob_a(coef_vec: np.ndarray):
    """Funkcja z drugiego zadania domowego
    Parameters:
    coef_vec : wetkor wartości wspołczynników
    Results:
    np.ndarray: macierza Frobeniusa o zadanych wartościach współczynników wielomianu 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(coef_vec , np.ndarray):
        
    #wektor coef_vect juz jest tym prawdziwym i wystarczy tylko zrobic flipa i dac na koniec i dac te jedynki 
        minus = np.negative(coef_vec)
        new = np.flip(minus)
        n = len(new)
        F = np.eye(n)
        first_col = np.zeros((n,1))
        F1 = np.hstack((first_col,F))
        add = np.append(new , [0])
        F1[n-1] = add 
        F2 = np.delete(F1 , -1 ,axis = 1 )   
        return F2
    else:
        return None 

    
def polly_from_egval(egval_vec: np.ndarray):
    """Funkcja z laboratorium 8
    Parameters:
    egval_vec: wetkor wartości własnych
    Results:
    np.ndarray: wektor współczynników wielomianu charakterystycznego
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(egval_vec, np.ndarray):
        return None
    else:      
        return np.polynomial.polynomial.polyfromroots(egval_vec)[::-1]

def L_inf(xr:Union[int, float, List, np.ndarray],x:Union[int, float, List, np.ndarray])-> float:
    """Obliczenie normy  L nieskończonośćg. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach biblioteki numpy.
    
    Parameters:
    xr (Union[int, float, List, np.ndarray]): wartość dokładna w postaci wektora (n,)
    x (Union[int, float, List, np.ndarray]): wartość przybliżona w postaci wektora (n,1)
    
    Returns:
    float: wartość normy L nieskończoność,
                                    NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(xr, (int, float)) and isinstance(x,(int,float)):
        return abs(xr - x)
    elif isinstance(xr, np.ndarray) and isinstance(x,np.ndarray) and xr.shape == x.shape:
       return max(abs(xr - x))
    elif isinstance(xr , List) and isinstance(x , List):
        return abs(max(xr) - max(x))
    else:
        return np.NaN 