import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

# 这个文件也是测SNR，但是使用的方差算法貌似可以防止数值错误，试一试
def prepare_data(url_trace, trace_name, url_data, data_name, n, loadData):
    m = [None] * 300
    v = [None] * 300
    M2 = [None] * 300
    # trace = np.load(url_trace + r"arrPart0.npy")
    trace = np.load(url_trace + trace_name.format(0))
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
        data = np.loadtxt(url_data + data_name.format(j), delimiter=',', dtype="int")
        print("此处用汉明重量，记得删除")
        data = [hammingWeight(label) for label in data]
        # print(data)
        trace = np.load(url_trace + trace_name.format(j))

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
        #if i_ == 1:
            #print("label", i_)
            #print("meanMatrix")
            #print(m[i_][29470])
    for j_ in loadData:
        noise.append(v[j_])
        #print("v[j_]")
        #print(v[j_])
    signal_var = np.var(signal, axis=0)

    # 这里好像不太一样/
    noise_mean = np.mean(noise, axis=0)
    return signal_var / noise_mean

# 求一个数的汉明重量
def hammingWeight(n):
    c = 0
    while n:
        c += 1
        n &= n - 1

    return c
def snr_function(n, url_trace, trace_name, url_data, data_name):
    plt.rcParams['figure.figsize'] = (12.0, 4.0)
    f, ax = plt.subplots(1, 1)
    ax.set_title('snr_traces')
    list = []
    for i in range(n):
        dataaaa = np.loadtxt(url_data + data_name.format(i), delimiter=',', dtype="int")
        print("此处变成hamming重量，记得恢复")
        dataaaa = [hammingWeight(label) for label in dataaaa]
        num = (np.unique(dataaaa)).tolist()

        list = list + num
    loadData = (np.unique(list))
    # 得到的loadData是*所有块中*去重后的输入数据


    result_data = prepare_data(url_trace, trace_name, url_data, data_name, n, loadData)
    np.save(url_trace + snr_save_file_name, result_data)
    ax.plot(result_data)
    plt.show()


if __name__ == '__main__':
    # 确定文件路径，名字，块数

    #url_trace = r"F:/weixinzeng32sh8/"
    #url_data = r"F:/weixinzeng32sh8/"

    url_trace = r"F:/weixinzeng_wave_filter_trace_16bit/"
    url_data = r"F:/weixinzeng_wave_filter_trace_16bit/"

    url_trace = r"D:/ChipWhisperer5_52/cw/home/portable/chipwhisperer/tutorials/courses/sca101/example_aes/"
    url_data = r"D:/ChipWhisperer5_52/cw/home/portable/chipwhisperer/tutorials/courses/sca101/example_aes/"

    data_name = "aaadata{0}.txt"
    data_name = "example_aes_inter_part{0}.txt"

    trace_name = "arrPart{0}.npy"
    trace_name = "example_aes_part{0}.npy"
    snr_save_file_name = "snr_data_better_var.npy"
    n = 250
    print("使用的块数", n)
    # 运行此函数
    snr_function(n, url_trace, trace_name, url_data, data_name)