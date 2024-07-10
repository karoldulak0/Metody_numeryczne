# -*- coding: utf-8 -*-

import pytest
import main
import pickle
import math
import numpy as np

from typing import Union, List, Tuple

expected = pickle.load(open('expected','rb'))

results_p_diff = expected['p_diff']
results_exponential = expected['exponential']
results_coskx1 = expected['coskx1']
results_coskx2 = expected['coskx2']

@pytest.mark.parametrize("n,c,result", results_p_diff)
def test_p_diff(n: int, c: Union[int, float], result):
    if np.any(np.isnan(result)):
        assert math.isnan(main.p_diff(n, c)), 'Spodziewany wynik: {0}, aktualny {1}. Błedy wejścia.'.format(result, main.p_diff(n, c))
    else:
        assert main.p_diff(n, c) == pytest.approx(result), 'Spodziewany wynik: {0}, aktualny {1}. Błędy implementacji.'.format(result, main.p_diff(n, c))


@pytest.mark.parametrize("x,n,result", results_exponential)
def test_exponential(x: Union[int, float], n: int, result):
    if np.any(np.isnan(result)):
        assert np.isnan(main.exponential(x, n)), 'Spodziewany wynik: {0}, aktualny {1}. Błedy wejścia.'.format(result, main.exponential(x, n))
    else:
        assert main.exponential(x, n) == pytest.approx(result), 'Spodziewany wynik: {0}, aktualny {1}. Błędy implementacji.'.format(result, main.exponential(x, n))


@pytest.mark.parametrize("k,x,result", results_coskx1)
def test_coskx1(k: int, x: Union[int, float], result):
    if np.any(np.isnan(result)):
        assert np.isnan(main.coskx1(k, x)), 'Spodziewany wynik: {0}, aktualny {1}. Błedy wejścia.'.format(result, main.coskx1(k, x))
    else:
        assert main.coskx1(k, x) == pytest.approx(result), 'Spodziewany wynik: {0}, aktualny {1}. Błędy implementacji.'.format(result, main.coskx1(k, x))


@pytest.mark.parametrize("k,x,result", results_coskx2)
def test_coskx2(k: int, x: Union[int, float], result):
    if np.any(np.isnan(result)):
        assert np.isnan(main.coskx2(k, x)), 'Spodziewany wynik: {0}, aktualny {1}. Błedy wejścia.'.format(result, main.coskx2(k, x))
    else:
        assert main.coskx2(k, x) == pytest.approx(result), 'Spodziewany wynik: {0}, aktualny {1}. Błędy implementacji.'.format(result, main.coskx2(k, x))

