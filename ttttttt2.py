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

#file_path = r"F:/weixinzeng_wave_filter_trace_16bit/"
file_path = r"F:/weixinzeng32sh8/"
# file_path = r"D:/ChipWhisperer5_52/cw/home/portable/chipwhisperer/jupyter/courses/sca101/traces/"
input_data_file_name = "aaadata{0}.txt"
# input_data_file_name = "lab3_3_zhongjianzhi_chaifen{0}.txt"
traces_file_name = "arrPart{0}.npy"
input_data = []
part = 60
for i in range(part):
    # 收集一下所有的输入数据
    input_data += np.loadtxt(file_path + input_data_file_name.format(i), delimiter=',', dtype="int").tolist()

print("此处把input_data变成hamming重量，记得恢复")
input_data = [hammingWeight(label) for label in input_data]

print(np.unique(input_data))
