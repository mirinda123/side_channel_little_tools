#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：sca
@File    ：zhoujiayun.py
@Author  ：suyang
@Date    ：2022/5/21 12:57
'''
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm, trange

# np.nanmax:排除nan值求最大
np.seterr(divide='ignore', invalid='ignore')
from numba import njit, jit

def getHW(n):
    ans = 0
    if n == 0:
        return 0
    while n > 0:
        n = n & (n - 1)
        ans += 1
    return ans
# all the attack need data matrix
def big_correlation_func(n, url_trace, trace_name,url_data,data_name):
    '''

    :param n: 变量的块数
    :param url_trace: 曲线集的路径
    :param url_data: 数据的路径
    :return:
    '''


    arr = np.load(url_trace + trace_name.format(0))
    # data = np.load(url_data + r"new_arrdata0.npy")
    Na = arr.shape[1]
    old_cov = np.zeros(Na)
    old_mean_data = 0
    old_mean_traces = np.zeros(Na)
    old_var_data = 0
    old_var_traces = np.zeros(Na)
    temp = 0
    for j in trange(n):
        data = np.loadtxt(url_data + data_name.format(j), delimiter=',', dtype="int")
        data = [getHW(label) for label in data]
        arr = np.load(url_trace + trace_name.format(j))
        for i in range(arr.shape[0]):
            new_mean_data = old_mean_data + (data[i] - old_mean_data) / (temp + 1)
            new_mean_traces = old_mean_traces + (arr[i] - old_mean_traces) / (temp + 1)

            new_var_data = old_var_data + ((data[i] - old_mean_data) * (data[i] - new_mean_data) - old_var_data) / (
                    temp + 1)
            new_var_traces = old_var_traces + (
                    (arr[i] - old_mean_traces) * (arr[i] - new_mean_traces) - old_var_traces) / (temp + 1)
            new_cov = (old_cov * temp + (data[i] - old_mean_data) * (arr[i] - new_mean_traces)) / (temp + 1)
            old_mean_data = new_mean_data
            old_mean_traces = new_mean_traces
            old_cov = new_cov
            old_var_traces = new_var_traces
            old_var_data = new_var_data
            temp = temp + 1
    correlation_result = old_cov / np.sqrt(old_var_traces * old_var_data)
    return correlation_result

if __name__ == '__main__':
    # 文件块数
    n = 300
    # url_trace = r"F:/weixinzeng32sh8/"
    # url_data = r"F:/weixinzeng32sh8/"

    # url_trace = r"F:/weixinzeng_wave_filter_trace_16bit/"
    # url_data = r"F:/weixinzeng_wave_filter_trace_16bit/"

    # url_trace = r"F:/another_CPU/5mhz_filter_8bit_new/"
    # url_data = r"F:/another_CPU/5mhz_filter_8bit_new/"

    url_trace = r"F:/another_CPU/5mhz_filter_16bit/"
    url_data = r"F:/another_CPU/5mhz_filter_16bit/"

    data_name = "aaadata{0}.txt"
    trace_name = "arrPart{0}.npy"
    snr_save_file_name = "xiang_guan_xing.npy"


    #运行函数且绘图
    f, ax = plt.subplots(1, 1)
    ax.set_title('hw_cpa_traces')
    result = big_correlation_func(n, url_trace, trace_name,url_data,data_name)
    ax.plot(result)

    np.save(url_trace + snr_save_file_name, result)
    plt.show()