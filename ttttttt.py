import math
import os
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np







print(np.linspace(5953,6116,30).astype(int).tolist())
print(np.linspace(6677,6846,30).astype(int).tolist())
print(np.linspace(7059,7225,30).astype(int).tolist())
print(np.linspace(7427,7570,30).astype(int).tolist())
print(np.linspace(7796,7972,30).astype(int).tolist())
print(np.linspace(8174,8343,30).astype(int).tolist())
print(np.linspace(8559,8719,30).astype(int).tolist())
print(np.linspace(8938,9037,30).astype(int).tolist())



def choosePoI(correlation_file_path, correlation_file_name, numPOIs, POIspacing):
    corr_trace = abs(np.load(correlation_file_path + correlation_file_name))[correlation_file_offset:]
    plt.plot(corr_trace)
    PoIs = []
    # Repeat until we have enough POIs
    for i in range(numPOIs):
        # Find the biggest peak and add it to the list of POIs
        nextPOI = corr_trace.argmax()
        PoIs.append(nextPOI)

        # 把周围的点清零
        # Make sure we don't go out of bounds
        poiMin = max(0, nextPOI - POIspacing)
        poiMax = min(nextPOI + POIspacing, len(corr_trace))
        for j in range(poiMin, poiMax):
            corr_trace[j] = 0
    return PoIs


correlation_file_path = r"C:/Users/jfj/Desktop/PoI/another_CPU/5mhz_filter_8bit_new/"
correlation_file_name = "xiang_guan_xing.npy"
correlation_file_offset = 5000
PoIs = choosePoI(correlation_file_path, correlation_file_name, 50,10)

print(PoIs)
plt.show()
