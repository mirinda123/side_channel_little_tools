import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

# 这个文件也是测SNR，但是使用的方差算法貌似可以防止数值错误，试一试
def prepare_data(url_trace, trace_name, url_data, data_name, n, loadData):
    m = [None] * 300
    v = [None] * 300
    M2 = [None] * 300
    # trace = np.load(url_trace + r"arrPart0.npy")
    trace = np.load(url_trace + r"arrPart0.npy")
    # count_temp = np.zeros(300, dtype=np.int32)

    # count_temp是统计每一个label的数量
    # 这里300是冗余了
    count_temp = np.zeros(300, dtype=np.int32)

    # 初始化m和v
    # i 是 1- 254
    # 我们需要0-255

    for i in loadData:
        m[i] = np.zeros(trace.shape[1])
        v[i] = np.zeros(trace.shape[1])
        M2[i] = np.zeros(trace.shape[1])
    # n是block数
    for j in trange(n):
        data = np.loadtxt(url_data + data_name + r"{0}.txt".format(j), delimiter=',', dtype="int")
        # print(data)
        trace = np.load(url_trace + trace_name + r"{0}.npy".format(j))

        for count, label in enumerate(data):
            count_temp[label] +=1
            # 枚举，count表示第几个，从0开始，到499

            # d[label].append(trace[count])

            # old_mean是一个向量
            # m[label]是一个向量

            delta = trace[count] - m[label]

            m[label] += (delta / count_temp[label])

            # 遍历每一条数据，在线计算均值和方差 Welford's online algorithm
            # m是均值
            # v是方差
            delta2 = trace[count] - m[label]
            M2[label] += delta * delta2
    for label in loadData:
        v[label] = M2[label] / count_temp[label]

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


def snr_function(n, url_trace, trace_name, url_data, data_name):
    plt.rcParams['figure.figsize'] = (12.0, 4.0)
    f, ax = plt.subplots(1, 1)
    ax.set_title('snr_traces')
    list = (np.unique(np.loadtxt(url_data + data_name + r"0.txt", delimiter=',', dtype="int"))).tolist()
    for i in range(n - 1):
        dataaaa = np.loadtxt(url_data + data_name + r"{0}.txt".format(i + 1), delimiter=',', dtype="int")
        num = (np.unique(dataaaa)).tolist()
        list = list + num
    loadData = (np.unique(list))
    # 得到的loadData是*所有块中*去重后的输入数据
    result_data = prepare_data(url_trace, trace_name, url_data, data_name, n, loadData)
    np.save(r"F:/tracexinzeng32sh8/snr_data_better_var.npy", result_data)
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
    snr_function(n, url_trace, trace_name, url_data, data_name)