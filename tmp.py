import torch
from torch import nn
from sklearn.decomposition import PCA
import numpy as np

# target output size of 5
t_list = [0.1, 0.3, 0.5, 0.7, 1, 2, 3]
for i in t_list:
    print(i)