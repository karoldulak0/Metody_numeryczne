import math 
import main

import numpy as np
import scipy

def cylinder_area(r:float,h:float):
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.
    
    Parameters:
    r (float): promień podstawy walca 
    h (float): wysokosć walca
    
    Returns:
    float: pole powierzchni walca 
    """
    if r < 0 or h < 0 or (not isinstance(r,float)and not (r,int))or (not isinstance(h,float)and not isinstance(h,int)):
        return np.NaN 
    return 2*math.pi *r *(r+h)

def fib(n:int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.
    
    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """ 
    if not isinstance(n,int) or n <= 0:
        return None 
    
    if n == 1:
        return np.array([1])
    else:
        result = [1,1]
        for i in range(2,n):
            result.append(result[-1]+result[-2])
        return np.array([result])       

   

def matrix_calculations(a:float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    a (float): wartość liczbowa 
    
    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    M = np.array([[a , 1 , -a ] , [ 0 , 1 , 1 ] , [-a , a , 1 ]])
    Mt = np.transpose(M)
    Mdet = np.linalg.det(M)
    if Mdet != 0 :
        
        return np.linalg.inv(M) , Mt , Mdet 
    else:
        
        return np.NaN , Mt , Mdet 

def custom_matrix(m:int, n:int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    

    if n <=0 or m <= 0 or type(n) != int or type(m) != int:
        return None 
    
    retMat = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if i > j:
                #indeks wiersza
                retMat[i,j] = i 
                
            else:
                #indeks kolumny 
                retMat[i,j] = j 


    return retMat