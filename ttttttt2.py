import math
import os
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np

def hammingWeight(n):
    c = 0
    while n:
        c += 1
        n &= n - 1

    return c

[for i in range]
