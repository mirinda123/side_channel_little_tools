import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import trange
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# 下面的代码用来生成数据
# 生成数据
import re

def to_one2(traces):

    # 判断是否是一个类型isinstance 不能用在numba里面
    # if not isinstance(traces, np.ndarray):
    #     raise TypeError("'data' should be a numpy ndarray.")
    new_traces = np.zeros((traces.shape[0], traces.shape[1]), dtype=float)
    a = np.mean(traces, axis=0)
    for t in range(traces.shape[0]):
        # a = np.mean(traces, axis=0)
        new_traces[t] = (traces[t] - a)
    return new_traces

sbox=(
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16)

def hammingWeight(n):
    c = 0
    while n:
        c += 1
        n &= n - 1

    return c


'''
for alun in range(0, 1000):
    f1 = open("F:/tracexinzeng32sh8/data45{0}.txt".format(alun), "r")
    fdata = open("F:/tracexinzeng32sh8/aaadata{0}.txt".format(alun), "a")
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

'''
'''

# 下面是测SNR


import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

def prepare_data(url_trace, trace_name, url_data, data_name, n,loadData):
    m = {}
    v = {}
    #trace = np.load(url_trace + r"arrPart0.npy")
    trace = np.load(url_trace + r"arrPart0.npy")
    #count_temp = np.zeros(300, dtype=np.int32)

    # count_temp是统计每一个label的数量
    # 这里300是冗余了
    count_temp = np.zeros(300, dtype=np.int32)

    # 初始化m和v
    for i in loadData:
        m[i] = np.zeros(trace.shape[1])
        v[i] = np.zeros(trace.shape[1])
    # n是多少个block
    for j in trange(n):
        data = np.loadtxt(url_data + data_name + r"{0}.txt".format(j), delimiter=',', dtype="int")
        # print(data)
        trace = np.load(url_trace +trace_name+ r"{0}.npy".format(j))
        for count, label in enumerate(data):
            # 枚举，count表示第几个，从0开始，到499

            # d[label].append(trace[count])

            # old_mean是一个向量
            # m[label]是一个向量
            old_mean = m[label]

            # 遍历每一条数据，在线计算均值和方差 Welford's online algorithm
            # m是均值
            # v是方差
            m[label] = m[label] + (trace[count] - m[label]) / (count_temp[label] + 1)
            if label == 1:
                print("trace:",trace[count])
                print("mean:",m[label])
            # sigma的平方
            v[label] = v[label] + ((trace[count] - old_mean) * (trace[count] - m[label]) - v[label]) / (
                    count_temp[label] + 1)
            count_temp[label] += 1
    signal = []
    noise = []
    for i_ in loadData:
        # signal_trace 存所有label的均值曲线
        signal.append(m[i_])
        #print("m[i_]")
        #print(m[i_])
    for j_ in loadData:
        noise.append(v[j_])
        #print("v[j_]")
        #print(v[j_])
    signal_var = np.var(signal, axis=0)

    # 这里好像不太一样/
    noise_mean = np.mean(noise, axis=0)
    return signal_var / noise_mean

def snr_function(n,url_trace,trace_name,url_data,data_name):
    plt.rcParams['figure.figsize'] = (12.0, 4.0)
    f, ax = plt.subplots(1, 1)
    ax.set_title('snr_traces')
    list = (np.unique(np.loadtxt(url_data + data_name+r"0.txt", delimiter=',', dtype="int"))).tolist()
    for i in range(n - 1):
        dataaaa = np.loadtxt(url_data + data_name + r"{0}.txt".format(i + 1), delimiter=',', dtype="int")
        num = (np.unique(dataaaa)).tolist()
        list = list + num
    loadData = (np.unique(list))
    # 得到的loadData是*所有块中*去重后的输入数据
    result_data = prepare_data(url_trace, trace_name, url_data, data_name, n,loadData)
    np.save(r"F:/tracexinzeng32sh8/snr_data_test.npy",result_data)
    ax.plot(result_data)
    plt.show()

if __name__ == '__main__':
    
    # 确定文件路径，名字，块数    
    
    url_trace = r"F:/tracexinzeng32sh8/"
    url_data = r"F:/tracexinzeng32sh8/"

    data_name = "aaadata"
    trace_name = "arrPart"
    n = 2
    # 运行此函数
    snr_function(n,url_trace,trace_name,url_data,data_name)


# 确定的几个POI
# x = 3134 4140 5160 6170 7190 8190 9220 10220 11240 12270 13300 14280  15310 16330 17330 18350 19360 20430 21390 23400 25430 27450  29470


'''
print("计算模板部分的代码没有问题")
print("如果有问题，可能在PCA部分或者攻击部分？")
print("下面的临时用汉明重量了！")
# file_path = r"F:/weixinzeng_wave_filter_trace_16bit/"
file_path = r"F:/weixinzeng32sh8/"
# file_path = r"D:/ChipWhisperer5_52/cw/home/portable/chipwhisperer/jupyter/courses/sca101/traces/"
input_data_file_name = "aaadata{0}.txt"
# input_data_file_name = "lab3_3_zhongjianzhi_chaifen{0}.txt"
traces_file_name = "arrPart{0}.npy"
# traces_file_name = "lab3_3_traces_chaifen{0}.npy"
# num_of_each_part = 500  # 每个part多少条曲线
num_of_each_part = 500
##########################
# 下面是进行模板攻击的参数 #
##########################
part = 7  # 选择多少个 part 进行建模
profiling_part_start_index = 0  # 建模曲线首先从哪一个part开始
num_of_class = 9  # 对于8bit HW是9类， 16bitHW是17类， 0-255有256类

##########################
# 下面是进行PCA的参数 #
##########################
pca_part = part  # 用多少 part 用来做 pca
pca_part_start_index = profiling_part_start_index    # 首先从哪一个part开始
pca_components = 5
use_pca = False  # 是否使用降维
# pca_method = ["full_traces", "mean_9", "mean_256", "mean_17"] 对应的下标
pca_method_select = 3

##########################
print("是否使用降维:", use_pca)
# PoI的下标
# 这个PoI对应的是 F:\tracexinzeng32sh8 256类
# PoIs = [3134, 4140, 5160, 6170, 7190, 8190, 9220, 10220,11240, 12270,13300 ,14280,15310,16330,17330,18350,19360,20430,21390,23400,25430,27450 ,29470]
# 这个PoI对应的是 F:\weixinzeng32sh8 256类
# PoIs = [1004, 1200, 1320, 1443, 1560, 1675, 1799, 1920]

# 这个PoI对应的是 F:\weixinzeng32sh8 9类，汉明重量 ，左边第一个尖峰没有用，左边第二个尖峰取的点多一些，取5个点吧,后面每个小尖峰都取2个点
# PoIs = [983, 985, 989, 994, 998, 1004, 1009, 1188, 1209, 1304, 1330, 1429, 1449, 1548, 1569, 1668, 1690, 1788, 1810,
#         1904, 1909, 1930]
# 这个PoI对应的是 F:\weixinzeng32sh8 9类，汉明重量 ，左边第一个尖峰没有用，后面每个尖峰取10个点
# F:\weixinzeng32sh8数据集是没有用滤波器，8位，但是没有0没有255
PoIs = [984, 989, 995, 1001, 1007, 1013, 1019, 1025, 1031, 1037,
1181, 1186, 1191, 1197, 1202, 1207, 1213, 1218, 1223, 1229,
1299, 1305, 1312, 1318, 1325, 1331, 1338, 1344, 1351, 1358,
1418, 1422, 1426, 1430, 1434, 1438, 1442, 1446, 1450, 1455,
1542, 1545, 1549, 1553, 1556, 1560, 1564, 1567, 1571, 1575,
1663, 1667, 1671, 1676, 1680, 1684, 1689, 1693, 1697, 1702,
1783, 1786, 1789, 1792, 1795, 1799, 1802, 1805, 1808, 1812,
1903, 1906, 1910, 1913, 1917, 1920, 1924, 1927, 1931, 1935]

# 这个PoI对应的是 F:\weixinzeng32sh8 9类，汉明重量 ，左边第一个尖峰没有用，后面每个尖峰取15个点
PoIs = [984, 987, 991, 995, 999, 1002, 1006, 1010, 1014, 1018, 1021, 1025, 1029, 1033, 1037,
1181, 1184, 1187, 1191, 1194, 1198, 1201, 1205, 1208, 1211, 1215, 1218, 1222, 1225, 1229,
1299, 1303, 1307, 1311, 1315, 1320, 1324, 1328, 1332, 1336, 1341, 1345, 1349, 1353, 1358,
1418, 1420, 1423, 1425, 1428, 1431, 1433, 1436, 1439, 1441, 1444, 1447, 1449, 1452, 1455,
1542, 1544, 1546, 1549, 1551, 1553, 1556, 1558, 1560, 1563, 1565, 1567, 1570, 1572, 1575,
1663, 1665, 1668, 1671, 1674, 1676, 1679, 1682, 1685, 1688, 1690, 1693, 1696, 1699, 1702,
1783, 1785, 1787, 1789, 1791, 1793, 1795, 1797, 1799, 1801, 1803, 1805, 1807, 1809, 1812,
1903, 1905, 1907, 1909, 1912, 1914, 1916, 1919, 1921, 1923, 1925, 1928, 1930, 1932, 1935]

# 这个PoI对应的是 F:\weixinzeng32sh8 9类，汉明重量 ，左边第一个尖峰没有用，后面每个尖峰取20个点
# PoIs = [984, 986, 989, 992, 995, 997, 1000, 1003, 1006, 1009, 1011, 1014, 1017, 1020, 1023, 1025, 1028, 1031, 1034, 1037,
# 1181, 1183, 1186, 1188, 1191, 1193, 1196, 1198, 1201, 1203, 1206, 1208, 1211, 1213, 1216, 1218, 1221, 1223, 1226, 1229,
# 1299, 1302, 1305, 1308, 1311, 1314, 1317, 1320, 1323, 1326, 1330, 1333, 1336, 1339, 1342, 1345, 1348, 1351, 1354, 1358,
# 1418, 1419, 1421, 1423, 1425, 1427, 1429, 1431, 1433, 1435, 1437, 1439, 1441, 1443, 1445, 1447, 1449, 1451, 1453, 1455,
# 1542, 1543, 1545, 1547, 1548, 1550, 1552, 1554, 1555, 1557, 1559, 1561, 1562, 1564, 1566, 1568, 1569, 1571, 1573, 1575,
# 1663, 1665, 1667, 1669, 1671, 1673, 1675, 1677, 1679, 1681, 1683, 1685, 1687, 1689, 1691, 1693, 1695, 1697, 1699, 1702,
# 1783, 1784, 1786, 1787, 1789, 1790, 1792, 1793, 1795, 1796, 1798, 1799, 1801, 1802, 1804, 1805, 1807, 1808, 1810, 1812,
# 1903, 1904, 1906, 1908, 1909, 1911, 1913, 1914, 1916, 1918, 1919, 1921, 1923, 1924, 1926, 1928, 1929, 1931, 1933, 1935]



# 这个PoI对应的是 F:\weixinzeng32sh8 9类，汉明重量 CPA 相关性，高于0.005的点左边第一个尖峰没有用
# PoIs = [1021, 1025, 1210, 1329, 1449, 1569, 1689, 1809, 1929]

# 这个PoI对应的是 weixinzeng_wave_filter_trace_16bit 每一个尖峰10个点 8个尖峰
# PoIs = [2448, 2464, 2481, 2497, 2514, 2530, 2547, 2563, 2580, 2597,
#         2946, 2956, 2966, 2976, 2986, 2996, 3006, 3016, 3026, 3037,
#         3184, 3200, 3216, 3232, 3248, 3264, 3280, 3296, 3312, 3328,
#         3546, 3555, 3564, 3573, 3582, 3591, 3600, 3609, 3618, 3627,
#         3847, 3855, 3863, 3872, 3880, 3889, 3897, 3906, 3914, 3923,
#         4146, 4154, 4163, 4171, 4180, 4188, 4197, 4205, 4214, 4223,
#         4446, 4454, 4462, 4471, 4479, 4488, 4496, 4505, 4513, 4522,
#         4746, 4754, 4763, 4772, 4781, 4790, 4799, 4808, 4817, 4826
#         ]

# 这个PoI对应的是 weixinzeng_wave_filter_trace_16bit （5hmz示波器）每一个尖峰10个点 8个尖峰

# PoIs = [2447, 2459, 2472, 2485, 2498, 2510, 2523, 2536, 2549, 2562,
# 2935, 2947, 2959, 2971, 2983, 2995, 3007, 3019, 3031, 3043,
# 3173, 3192, 3211, 3230, 3249, 3268, 3287, 3306, 3325, 3345,
# 3545, 3555, 3566, 3576, 3587, 3597, 3608, 3618, 3629, 3640,
# 3836, 3846, 3856, 3866, 3876, 3886, 3896, 3906, 3916, 3926,
# 4140, 4151, 4163, 4175, 4186, 4198, 4210, 4221, 4233, 4245,
# 4428, 4439, 4450, 4461, 4472, 4483, 4494, 4505, 4516, 4528,
# 4730, 4741, 4752, 4764, 4775, 4786, 4798, 4809, 4820, 4832]

# 这个PoI对应的是 weixinzeng_wave_filter_trace_16bit （5hmz示波器）每一个尖峰15个点 8个尖峰
# PoIs = [2447, 2455, 2463, 2471, 2479, 2488, 2496, 2504, 2512, 2520, 2529, 2537, 2545, 2553, 2562,
# 2935, 2942, 2950, 2958, 2965, 2973, 2981, 2989, 2996, 3004, 3012, 3019, 3027, 3035, 3043,
# 3173, 3185, 3197, 3209, 3222, 3234, 3246, 3259, 3271, 3283, 3295, 3308, 3320, 3332, 3345,
# 3545, 3551, 3558, 3565, 3572, 3578, 3585, 3592, 3599, 3606, 3612, 3619, 3626, 3633, 3640,
# 3836, 3842, 3848, 3855, 3861, 3868, 3874, 3881, 3887, 3893, 3900, 3906, 3913, 3919, 3926,
# 4140, 4147, 4155, 4162, 4170, 4177, 4185, 4192, 4200, 4207, 4215, 4222, 4230, 4237, 4245,
# 4428, 4435, 4442, 4449, 4456, 4463, 4470, 4478, 4485, 4492, 4499, 4506, 4513, 4520, 4528,
# 4730, 4737, 4744, 4751, 4759, 4766, 4773, 4781, 4788, 4795, 4802, 4810, 4817, 4824, 4832]

# 这个PoI对应的是 AES D:\ChipWhisperer5_52\cw\home\portable\chipwhisperer\jupyter\courses\sca101\traces  lab3_3_yihuohoudezhiPart0.npy 的脚本自动挑选出来的
# PoIs = [1313, 1318, 1341, 2165, 1661]

# full 是 对全体 pca_part * num_of_each_part 条曲线做PCA_fit
# mean_9 是用汉明重量，对9条平均曲线做PCA_fit
# mean_256 是对256条平均曲线做PCA_fit
scale= StandardScaler()
pca_method = ["full_traces", "mean_9", "mean_256", "mean_17"]
if use_pca == True:
    # 如果使用降维，则替换PoI
    PoIs = [i for i in range(pca_components)]  # 这个PoI对应的是用PCA，pca_component= 3, PoIs = [0,1,2]相当于每一个点都取
    assert len(PoIs) == pca_components

# PoI的个数
num_of_PoIs = len(PoIs)
tempTraces = np.load(file_path + traces_file_name.format(0))
if use_pca == True:
    pca = PCA(n_components=pca_components)
    print("开始降维")
    if pca_method[pca_method_select] == "full_traces":
        # 暂时存放用来pca的所有曲线
        temp_for_pca = np.zeros((pca_part * num_of_each_part, tempTraces.shape[1]))
        # 遍历part
        for i in trange(pca_part):
            # 把用来PCA的曲线都放到   temp_for_pca 里面
            t = np.load(file_path + traces_file_name.format(i + pca_part_start_index))
            t = scale.fit_transform(t)
            temp_for_pca[i * num_of_each_part:(i + 1) * num_of_each_part] = t
        # 全部放完了，做拟合
        pca.fit(temp_for_pca)
    if pca_method[pca_method_select] == "mean_9" or pca_method[pca_method_select] == "mean_17" or pca_method[pca_method_select] == "mean_256":
        if pca_method[pca_method_select] == "mean_9":
            pca_classes = 9
        elif pca_method[pca_method_select] == "mean_17":
            pca_classes = 17
        else:
            pca_classes = 256
        # 此时的temp_for_pca里面存的是均值, 相当于meanMatrix
        # 注意可能有0，要去除0！！！！！
        temp_for_pca = np.zeros((pca_classes, tempTraces.shape[1]))
        # 记录每一个label有多条曲线
        pca_count = [0] * pca_classes
        for i in trange(pca_part):
            label_temp = np.loadtxt(file_path + input_data_file_name.format(i + pca_part_start_index), delimiter=' ',
                                    dtype="int")

            # 如果选择的是mean_9或mean_17就计算hamming重量
            if pca_method[pca_method_select] == "mean_9" or pca_method[pca_method_select] == "mean_17":
                label_temp = [hammingWeight(label) for label in label_temp]

            trace_temp = np.load(file_path + traces_file_name.format(i + pca_part_start_index))
            trace_temp = scale.fit_transform(trace_temp)
            print("均值PCA时零值需要剔除吗")
            # 遍历一个Part中所有曲线
            for index, label in enumerate(label_temp):
                pca_count[label] += 1
                # # 算出新的均值
                #         meanMatrix[label] = meanMatrix[label] + (trace - meanMatrix[label]) / (count[label])
                temp_for_pca[label] = temp_for_pca[label] + (trace_temp[index] - temp_for_pca[label]) / (
                pca_count[label])

        # 到这里计算出了均值
        pca.fit(temp_for_pca)

# 初始化均值向量
meanMatrix = [None] * num_of_class  # 对于每个类，存放每个PoI处的均值
old_mean = [None] * num_of_class
# 初始化协方差矩阵
covMatrix = [None] * num_of_class
# 初始化count ，用来记录每一个label下曲线的数目
count = [None] * num_of_class
for i in range(num_of_class):
    meanMatrix[i] = np.zeros(num_of_PoIs)
for i in range(num_of_class):
    old_mean[i] = np.zeros(num_of_PoIs)
for i in range(num_of_class):
    covMatrix[i] = np.zeros((num_of_PoIs, num_of_PoIs))
for i in range(num_of_class):
    count[i] = 0



print("开始建模")
for i in trange(part):
    input_data = np.loadtxt(file_path + input_data_file_name.format(i + profiling_part_start_index), delimiter=' ',
                            dtype="int")  # 读取一个part的输入数据
    input_data = [hammingWeight(label) for label in input_data]  # 转化成汉明重量

    # 从profiling_part_start_index开始，读取这个 part 的曲线
    traces = np.load(file_path + traces_file_name.format(i + profiling_part_start_index))

    #traces = to_one2(traces)    # 标准化一下
    traces = scale.fit_transform(traces) 
    if use_pca == True:
        traces = pca.transform(traces)  # 使用PCA转化一下曲线
        assert traces.shape[1] == num_of_PoIs
    assert len(input_data) == traces.shape[0]

    # 下面这个写法没有问题
    traces = traces[:, PoIs]  # 处理曲线，只选取这些PoI。如果使用PCA，那么PoI是[0,1,2],相当于选取转化后的所有点

    # 遍历这个 part 中所有的曲线
    # online 计算 cov
    # index 是下标
    for index, label in enumerate(input_data):
        count[label] += 1  # label 对应的曲线数量+1
        trace = traces[index]  # 取出这条曲线
        old_mean[label] = meanMatrix[label]  # 暂存一下旧的均值

        # 在线计算均值和方差 Welford's online algorithm
        # m是均值
        meanMatrix[label] = meanMatrix[label] + (trace - meanMatrix[label]) / (count[label])  # 算出新的均值
        zanting = 1
        # 在线计算协方差
        for i in range(num_of_PoIs):
            for j in range(num_of_PoIs):
                x = trace[i]
                y = trace[j]

                dx = x - old_mean[label][i]

                # new meanx
                # meanx += dx / count[label]

                # new meany
                # meany += (y - meany) / count[label]
                # C =( C * (n-1) +  dx * (y - meany) ) / n
                covMatrix[label][i][j] += dx * (y - meanMatrix[label][j])
# 所有 part 遍历完成之后，最终求出协方差
for label in range(num_of_class):
    for i in range(num_of_PoIs):
        for j in range(num_of_PoIs):
            covMatrix[label][i][j] = covMatrix[label][i][j] / (count[label] - 1)

# 求协方差这些关键的算法验证了，没有问题!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

########################################


########################################################



# 下面是对多条曲线进行攻击,但是每个label只使用一条曲线进行攻击，计算所有攻击的成功率
P_k = np.zeros(num_of_class)  # num_of_class 种猜测的分数

attack_part = 8  # 攻击曲线使用多少个part
attack_part_start_index = 600  # 从第几个part开始
success_num = 0  # 记录有多少条攻击成功了
count_of_used_traces = 0  # 记录有多少条label为target_label的曲线被使用了
success_rate_list = []  # 记录每一次攻击的成功率
print("开始攻击，对多条曲线, 求成功率")
for i in trange(attack_part):
    # 从 attack_part_start_index 开始，依次读取一个part的曲线
    traces = np.load(file_path + traces_file_name.format(i + attack_part_start_index))
    # traces = to_one2(traces)    # 标准化一下
    traces = scale.fit_transform(traces)
    # 降维
    if use_pca == True:
        traces = pca.transform(traces)
        assert traces.shape[1] == num_of_PoIs
    input_data = np.loadtxt(file_path + input_data_file_name.format(i + attack_part_start_index), delimiter=',',
                            dtype="int").tolist()
    input_data = [hammingWeight(label) for label in input_data]

    assert traces.shape[0] == len(input_data)


    for j in range(traces.shape[0]):    # 遍历这个part的每一条曲线

        count_of_used_traces += 1  # 使用的曲线数目 +1
        a = [traces[j][PoIs[i]] for i in range(len(PoIs))]  # 取出PoI所对应的点，重组成a
        for label in range(num_of_class):  # 测试每一个模板 ，本来这里写的是 for label in input_data_unique
            # 有边界情况：协方差矩阵为nan，说明没有这个模板, 说明输入数据没有这个
            # 如果为0，说明输入数据只有1条曲线有这个，没法算出方差，方差为0
            if np.isnan(covMatrix[label][0][0]) or np.isnan(meanMatrix[label][0]) or covMatrix[label][0][0] == 0 or \
                    meanMatrix[label][0] == 0:
                print("label为" + str(label) + "的模板不存在,建模曲线数量不够")
                P_k[label] = float("-inf")  # 不存在的话，这个模板对应的分数就是负无穷
                continue
            print(meanMatrix[label].shape)
            print(covMatrix[label].shape)
            rv = multivariate_normal(meanMatrix[label], covMatrix[label],allow_singular=True)  # 选择 label 所对应的模型
            p_kj = rv.pdf(a)  # 带入公式
            # 注意 这里不是+=了,不是求累计的分数
            P_k[label] = np.log(p_kj)  # 每一个模板的分数放进P_k
        # 如果最右边的数（猜测的密钥中分数最高的 所对应的下标） == label
        if P_k.argsort()[-1] == input_data[j]:  # input_data[j] 是这条曲线对应的 label，已经经过汉明重量的处理了
            success_num += 1
        print("成功率", success_num / count_of_used_traces)
        success_rate_list.append(success_num / count_of_used_traces)

plt.plot(success_rate_list)
plt.show()





'''
# 下面这个验证是正确的！！！！
# 下面是对一条曲线进行攻击
# 累加256种密钥猜测的分数
# 适用于chipwhisperer_lab3_3官方数据集用来验证用
# 原本是数据集是2500条，被我拆分成了500条一个part
attack_part = 1  # 攻击曲线使用多少个part
attack_part_start_index = 4  # 从第几个part开始

true_key  = 0x2b
P_k = np.zeros(256)

for i in trange(attack_part):
    atkTraces = np.load(file_path + traces_file_name.format(i + attack_part_start_index))
    # 取最后500个
    atkPText = np.load(r"D:/ChipWhisperer5_52/cw/home/portable/chipwhisperer/jupyter/courses/sca101/traces/lab3_3_textin.npy")[-500:]
    for j in range(len(atkTraces)):
        # Grab key points and put them in a small matrix
        a = [atkTraces[j][PoIs[i]] for i in range(len(PoIs))]

        # Test each key
        for k in range(256):
            # Find HW coming out of sbox

            HW = hammingWeight(sbox[atkPText[j][0] ^ k])
            input = np.loadtxt(file_path + input_data_file_name.format(i + attack_part_start_index), delimiter=',',
                                    dtype="int").tolist()

            if k == true_key:
                assert sbox[atkPText[j][0] ^ k] == input[j]
            # Find p_{k,j}
            rv = multivariate_normal(meanMatrix[HW], covMatrix[HW])
            p_kj = rv.pdf(a)

            # Add it to running total
            P_k[k] += np.log(p_kj)

        # Print our top 5 results so far
        # Best match on the right
        print(P_k.argsort()[-5:])


'''