import numpy as np

# 下面的代码用来生成数据
# 生成数据
import re

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


# 下面是测SNR

'''
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

            # sigma的平方
            v[label] = v[label] + ((trace[count] - old_mean) * (trace[count] - m[label]) - v[label]) / (
                    count_temp[label] + 1)
            count_temp[label] += 1
    signal = []
    noise = []
    for i_ in loadData:
        # signal_trace 存所有label的均值曲线
        signal.append(m[i_])
    for j_ in loadData:
        noise.append(v[j_])
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
    np.save(r"F:/tracexinzeng32sh8/snr_data.npy",result_data)
    ax.plot(result_data)
    plt.show()

if __name__ == '__main__':
    
    # 确定文件路径，名字，块数    
    
    url_trace = r"F:/tracexinzeng32sh8/"
    url_data = r"F:/tracexinzeng32sh8/"

    data_name = "aaadata"
    trace_name = "arrPart"
    n = 1000
    # 运行此函数
    snr_function(n,url_trace,trace_name,url_data,data_name)

'''
# 确定的几个POI
# x = 3134 4140 5160 6170 7190 8190 9220 10220 11240 12270 13300 14280  15310 16330 17330 18350 19360 20430 21390 23400 25430 27450  29470



# 下面是进行模板攻击

part = 10
# 首先对曲线进行分类
num_of_class = 256

# 0 - 255
# 用来给曲线分类
category = [[] for _ in range(num_of_class)]
for i in range(part):

    # 读取一个part的输入数据
    input_data = np.loadtxt("F:/tracexinzeng32sh8/aaadata{0}.txt".format(i), delimiter=' ', dtype="int")

    # 一个part的曲线
    traces = np.load("F:/tracexinzeng32sh8/arrPart{0}.npy".format(i))


    # 遍历这个part中所有的label
    for index, label in enumerate(input_data):
        category[label].append(traces[index].tolist())

print(category)
print(category[1])
category = np.array(category)
print(category.shape)