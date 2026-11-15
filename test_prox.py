#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pytest

from prox import soft_threshold, prox


def test_soft_threshold_positive():
    result = soft_threshold(5, 2)
    assert result == 3


def test_soft_threshold_negative():
    result = soft_threshold(-5, 2)
    assert result == -3


def test_soft_threshold_inside_threshold():
    result = soft_threshold(1, 2)
    assert result == 0


def test_soft_threshold_boundary():
    assert soft_threshold(2, 2) == 0
    assert soft_threshold(-2, 2) == 0


def test_soft_threshold_vector():
    x = np.array([-3, -0.5, 0.8, 2, 4])

    result = soft_threshold(x, 1)

    expected = np.array([-2, 0, 0, 1, 3])

    np.testing.assert_allclose(result, expected)


def test_prox():
    v = np.array([-3, -0.5, 0.8, 2, 4])

    result = prox(v, alpha=0.5, lam=2)

    expected = np.array([-2, 0, 0, 1, 3])

    np.testing.assert_allclose(result, expected)


def test_prox_zero_lambda():
    x = np.array([-2, 0, 3])

    result = prox(x, alpha=0.5, lam=0)

    np.testing.assert_allclose(result, x)


def test_negative_threshold():
    with pytest.raises(ValueError):
        soft_threshold(3, -1)


def test_invalid_alpha():
    with pytest.raises(ValueError):
        prox(3, alpha=0, lam=1)


# In[ ]:




