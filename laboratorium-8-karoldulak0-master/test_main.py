# -*- coding: utf-8 -*-

import pytest
import main
import pickle
import math
import numpy as np
import random

from typing import Union, List, Tuple

np.random.seed(24122022)
random.seed(24122022)
expected = pickle.load(open('expected','rb'))

result_egval_vec = expected['egval_vec_test']
result_coef_vec = expected['coef_vec_test']


@pytest.mark.parametrize("egval_vec,result", result_egval_vec)
def test_random_matrix_by_egval(egval_vec:np.ndarray, result):
    if result is None:
        assert main.random_matrix_by_egval(egval_vec) is None, 'Spodziewany wynik: {0}, aktualny {1}. Błedy wejścia.'.format(result, main.random_matrix_by_egval(egval_vec))
    else:
        A = main.random_matrix_by_egval(egval_vec)
        assert A == pytest.approx(result), 'Spodziewany wynik: {0}, aktualny {1}. Błedy wejścia.'.format(result, main.random_matrix_by_egval(egval_vec))


@pytest.mark.parametrize("coef_vec,result", result_coef_vec)
def test_frob_a(coef_vec:np.ndarray, result):
    if result is None:
        assert main.frob_a(coef_vec) is None, 'Spodziewany wynik: {0}, aktualny {1}. Błedy wejścia.'.format(result, main.frob_a(coef_vec))
    else:
        A = main.frob_a(coef_vec)
        assert A == pytest.approx(result), 'Spodziewany wynik: {0}, aktualny {1}. Błedy wejścia.'.format(result, main.frob_a(coef_vec))
