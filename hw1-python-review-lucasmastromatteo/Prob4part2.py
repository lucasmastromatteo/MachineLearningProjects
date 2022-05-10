import numpy as np  # Used to perform efficient (and convenient) array and matrix operations.
from sklearn import datasets  # Used to load standard datasets.
from matplotlib import pyplot as plt  # Used to create plots.
import math  # Used for trigonometric functions, log, pow, etc.

one = np.arange(2,7,1)
print(one)
two = np.ones((4,4))
print(two)
three = np.eye(6)
print(three)
test = np.array([[1,2,3],[0,0,0],[3,2,1]])
print(test)
four = np.sum(test,axis=0)
print(four)
A = np.array([[1,1],[2,2]])
B = np.array([[1,0],[0,1]])
C = np.matmul(np.transpose(A),B)
print(C)
col1 = np.array([[0],[0],[0],[0]])
col2 = np.array([[1],[1],[1],[1]])
six = np.hstack((col1,col2))
print(six)

