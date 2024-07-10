import pickle
from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import scipy as sp
from scipy import linalg


def spare_matrix_Abt(m: int,n: int):
    """Funkcja tworząca zestaw składający się z macierzy A (m,n), wektora b (m,)  i pomocniczego wektora t (m,) zawierających losowe wartości
    Parameters:
    m(int): ilość wierszy macierzy A
    n(int): ilość kolumn macierzy A
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,n) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m,int) and m > 0 and isinstance(n,int) and n >0: 
        temp = np.linspace(0, 1, m) #tworze wektor t od 0 do 1 o ilosci elementow m 
        t = np.transpose(temp) #aby był size = (m,)
        Atemp = np.vander(t, n) #macierz vandera - pierwsza kolumna t^n-1 druga t^n-2 itd 
        b = np.cos(4*t)
        A = np.fliplr(Atemp) #odwrocenie wzdluz srodkowej osi 
        return A , b 
    
    else: 
        return None



def square_from_rectan(A: np.ndarray, b: np.ndarray):
    """Funkcja przekształcająca układ równań z prostokątną macierzą współczynników na kwadratowy układ równań. Funkcja ma zwrócić nową macierz współczynników  i nowy wektor współczynników
    Parameters:
      A: macierz A (m,n) zawierająca współczynniki równania
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (n,n) i wektorem (n,)
             Jeżeli dane wejściowe niepoprawne funkcja zwraca None
     """

    if isinstance(A, np.ndarray) and isinstance(b, np.ndarray):
      A_transpose = np.transpose(A)
      l = np.dot(A_transpose,A)
      r = np.dot(A_transpose,b)
      return l, r
    else: 
      return None



def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray):
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

      Parameters:
      A: macierz A (m,n) zawierająca współczynniki równania
      x: wektor x (n,) zawierający rozwiązania równania
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania

      Results:
      (float)- wartość normy residuom dla podanych parametrów
      """
    if isinstance(A, np.ndarray) and isinstance(x , np.ndarray) and isinstance(b , np.ndarray):
      if A.shape[0] == b.shape[0] and A.shape[1] == x.shape[0]:
        Ax = np.dot(A,x)
        r = b - Ax
      return np.linalg.norm(r) 
    else:
      return None