# -*- coding: utf-8 -*-
"""Leaky ReLU.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I5GwjSx-KcoVqQgAcJwT3kxHHpBMRpXx
"""

import numpy as np

def Leaky_ReLU(x):
  if x > 0:
    return x
  return 0.01 * x


test = np.array([-10, 0, 10])
for i in test:
  print(f'Value for x = {i} is Leaky ReLU: {Leaky_ReLU(i)}')