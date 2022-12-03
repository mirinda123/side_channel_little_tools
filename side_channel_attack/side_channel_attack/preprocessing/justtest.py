from  pre_traces import to_one2
import numpy as np
import matplotlib.pyplot as plt

import re

for alun in range(0, 1000):
    f1 = open(r"F:\another_CPU\5mhz_filter_16bit\data45{0}.txt".format(alun), "r")
    fdata = open(r"F:\another_CPU\5mhz_filter_16bit\aaadata{0}.txt".format(alun), "a")
    # fxiyj = open("aaaxiyj{0}.txt".format(alun), "a")
    f1list = f1.readlines()
    f1str = f1list[0]
    f1str = f1str.lstrip()
    data = []
    liststr = re.findall(r'\d+', f1str)
    for i in range(len(liststr)):
        if (i % 45) < 32:
            data.append(int(liststr[i]))
        if (i + 1) % 45 == 0:
            outputdata = data[0]

            fdata.write(str(outputdata))
            fdata.write('\n')
            data = []

    f1.close()

    fdata.close()





