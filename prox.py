#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


def soft_threshold(x, threshold):
    """
    Toán tử ngưỡng mềm (Soft Thresholding).

    S_threshold(x) = sign(x) * max(|x| - threshold, 0)

    Parameters
    ----------
    x : scalar hoặc array
        Giá trị đầu vào.
    threshold : float
        Ngưỡng không âm.

    Returns
    -------
    scalar hoặc numpy.ndarray
        Kết quả sau khi áp dụng ngưỡng mềm.
    """
    if threshold < 0:
        raise ValueError("threshold phải không âm")

    x = np.asarray(x)

    result = np.sign(x) * np.maximum(
        np.abs(x) - threshold, 0
    )

    if result.ndim == 0:
        return result.item()

    return result


def prox(v, alpha, lam):
    """
    Toán tử proximal của hàm g(x) = lam * ||x||_1.

    prox_{alpha*g}(v) = soft_threshold(v, alpha * lam)

    Parameters
    ----------
    v : scalar hoặc array
        Vector đầu vào.
    alpha : float
        Tham số bước, alpha > 0.
    lam : float
        Hệ số điều chuẩn, lam >= 0.

    Returns
    -------
    scalar hoặc numpy.ndarray
        Kết quả của toán tử proximal.
    """
    if alpha <= 0:
        raise ValueError("alpha phải lớn hơn 0")

    if lam < 0:
        raise ValueError("lam phải không âm")

    return soft_threshold(v, alpha * lam)


# In[ ]:




