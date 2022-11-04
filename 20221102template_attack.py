import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import trange
from sklearn.decomposition import PCA
# 下面的代码用来生成数据
# 生成数据
import re


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

print("下面的临时用汉明重量了！")
file_path = r"F:/weixinzeng32sh8/"
input_data_file_name = "aaadata{0}.txt"
traces_file_name = "arrPart{0}.npy"
# 每个part多少条曲线
num_of_each_part = 500

##########################
# 下面是进行模板攻击的参数 #
##########################
# 选择多少个part进行建模
part = 500
# 建模曲线首先从哪一个part开始
profiling_part_start_index = 0
# 首先对曲线进行分类，0-255有256类
# HW是9类
num_of_class = 9
##########################




##########################
# 下面是进行PCA的参数 #
##########################
# 用来做pca的part
pca_part = 100
pca_part_start_index = 0
pca_components = 3
#是否使用降维
use_pca = False
pca_method_select = 2
print("是否使用降维:")
print(use_pca)
##########################

# PoI的下标
# 这个PoI对应的是 F:\tracexinzeng32sh8 256类
#PoIs = [3134, 4140, 5160, 6170, 7190, 8190, 9220, 10220,11240, 12270,13300 ,14280,15310,16330,17330,18350,19360,20430,21390,23400,25430,27450 ,29470]
# 这个PoI对应的是 F:\weixinzeng32sh8 256类
#PoIs = [1004, 1200, 1320, 1443, 1560, 1675, 1799, 1920]

# 这个PoI对应的是 F:\weixinzeng32sh8 9类，汉明重量 ，左边第一个尖峰没有用，左边第二个尖峰取的点多一些，取5个点吧,后面每个小尖峰都取2个点
PoIs = [983, 985, 989, 994, 998, 1004, 1009, 1188, 1209, 1304, 1330, 1429, 1449, 1548, 1569, 1668, 1690, 1788, 1810, 1904, 1909, 1930]

# 这个PoI对应的是 F:\weixinzeng32sh8 9类，汉明重量 CPA 相关性，高于0.005的点左边第一个尖峰没有用
PoIs = [1021, 1025, 1210, 1329, 1449, 1569, 1689, 1809, 1929]
# 这个PoI对应的是用PCA，pca_component= 3的话


# full 是 对全体 pca_part * num_of_each_part 条曲线做PCA_fit
# mean_9 是用汉明重量，对9条平均曲线做PCA_fit
# mean_256 是对256条平均曲线做PCA_fit




pca_method = ["full_traces","mean_9","mean_256"]
if use_pca == True:
    # 如果使用降维，则替换PoI
    PoIs = [0, 1, 2]
    assert len(PoIs) == pca_components

#PoI的个数
num_of_PoIs = len(PoIs)
tempTraces = np.load(file_path +traces_file_name.format(0))
if use_pca == True:
    pca = PCA(n_components=pca_components)
    print("开始降维")
    if pca_method[pca_method_select] == "full_traces":
        # 存放用来pca的所有曲线
        temp_for_pca = np.zeros((pca_part * num_of_each_part, tempTraces.shape[1]))
        # 遍历part
        for i in trange(pca_part):
            # 把用来PCA的曲线都放到   temp_for_pca 里面
            temp_for_pca[i * num_of_each_part:(i+1) * num_of_each_part] = np.load(file_path + traces_file_name.format(i + pca_part_start_index))
        # 全部放完了，做拟合
        pca.fit(temp_for_pca)
    if pca_method[pca_method_select] == "mean_9" or pca_method[pca_method_select] == "mean_256":
        if pca_method[pca_method_select] == "mean_9":
            pca_classes = 9
        else:
            pca_classes = 256
        # 此时的temp_for_pca里面存的是均值
        # 注意可能有0，要去除0！！！！！
        temp_for_pca = np.zeros((pca_classes , tempTraces.shape[1]))
        # 记录每一个label有多条曲线
        pca_count = [0] * pca_classes
        for i in trange(pca_part):
            label_temp = np.loadtxt(file_path + input_data_file_name.format(i + pca_part_start_index), delimiter=' ', dtype="int")

            # 如果选择的是mean_9就计算hamming重量
            if pca_method[pca_method_select] == "mean_9":
                label_temp = [hammingWeight(label) for label in label_temp]

            trace_temp = np.load(file_path + traces_file_name.format(i + pca_part_start_index))

            # 遍历一个Part中所有曲线
            for index, label in enumerate(label_temp):
                pca_count[label] +=1
                # # 算出新的均值
                #         meanMatrix[label] = meanMatrix[label] + (trace - meanMatrix[label]) / (count[label])
                temp_for_pca[label_temp] = temp_for_pca[label] + (trace_temp[index] - temp_for_pca[label]) / (pca_count[label])

        # 到这里计算出了均值

        assert temp_for_pca[0][0] == 0
        assert temp_for_pca[-1][0] == 0

        temp_for_pca = temp_for_pca[1:-1]
        # 实际上只有类数-2，因为没有采集0，没有采集255
        assert temp_for_pca.shape[0] == pca_classes-2
        pca.fit(temp_for_pca)




# 0 - 255
# 对于每个类，存放每个PoI处的均值
# 初始化均值向量
meanMatrix = [None] * num_of_class
old_mean = [None] * num_of_class
# 初始化协方差矩阵
covMatrix = [None] * num_of_class
#初始化count ，用来记录每一个label下曲线的数目
count = [None] * num_of_class
for i in range(num_of_class):
    meanMatrix[i] = np.zeros(num_of_PoIs)
for i in range(num_of_class):
    old_mean[i] = np.zeros(num_of_PoIs)
for i in range(num_of_class):
    covMatrix[i] = np.zeros((num_of_PoIs, num_of_PoIs))
for i in range(num_of_class):
    count[i] = 0


input_data= []
for i in range(part):
    # 收集一下所有的输入数据
    input_data +=  np.loadtxt(file_path + input_data_file_name.format(i), delimiter=',', dtype="int").tolist()


print("此处把input_data变成hamming重量，记得恢复")
input_data = [hammingWeight(label) for label in input_data]


# 对所有输入数据去重一下
input_data_unique =  np.unique(input_data)

print("开始建模")
for i in trange(part):
    # 读取一个part的输入数据
    input_data = np.loadtxt(file_path + input_data_file_name.format(i + profiling_part_start_index), delimiter=' ', dtype="int")
    input_data = [hammingWeight(label) for label in input_data]
    # 这个input_data没有问题
    # 一个part的曲线
    traces = np.load(file_path +traces_file_name.format(i + profiling_part_start_index))
    if use_pca == True:
        traces = pca.transform(traces)
    assert len(input_data) == traces.shape[0]
    # 处理曲线，只选取这些PoI
    traces = traces[:, PoIs]


    # 遍历这个part中所有的曲线
    # online 计算 cov
    for index, label in enumerate(input_data):

        #label对应的数量+1
        count[label] += 1

        # 取出这条曲线
        trace = traces[index]

        # 暂存一下旧的均值
        old_mean[label] = meanMatrix[label]

        # 在线计算均值和方差 Welford's online algorithm
        # m是均值

        # 算出新的均值
        meanMatrix[label] = meanMatrix[label] + (trace - meanMatrix[label]) / (count[label])

        #在线计算协方差
        for i in range(num_of_PoIs):
            for j in range(num_of_PoIs):
                x = trace[i]
                y = trace[j]

                dx = x - old_mean[label][i]

                # new meanx
                #meanx += dx / count[label]

                # new meany
                # meany += (y - meany) / count[label]
                # C =( C * (n-1) +  dx * (y - meany) ) / n
                covMatrix[label][i][j] += dx * (y - meanMatrix[label][j])

    #计算均值

# 所有part遍历完成之后，最终求出协方差
for label in input_data_unique:
    for i in range(num_of_PoIs):
        for j in range(num_of_PoIs):
            covMatrix[label][i][j] = covMatrix[label][i][j] / (count[label]-1)

# 求协方差这些关键的算法验证了，没有问题


########################################

'''
# 下面是开始攻击，对多个曲线,取出相同label的曲线，进行攻击

# 256种猜测的分数
P_k = np.zeros(num_of_class)

# 要对哪一个label进行攻击，我们需要pick出这些曲线。因为源文件中的trace有不同的label
target_label = 4
print("攻击的应该是" + str(target_label))
# 攻击曲线使用多少个part
attack_part = 50
# 从第几个part开始
attack_part_start_index = 500

#记录有多少条label为target_label的曲线被使用了
count_of_used_traces = 0
print("开始攻击，对多条曲线")
for i in trange(attack_part):

    # 一个part的曲线
    traces = np.load(file_path + traces_file_name.format(i + attack_part_start_index))
    input_data = np.loadtxt(file_path + input_data_file_name.format(i + attack_part_start_index), delimiter=',', dtype="int").tolist()
    input_data = [hammingWeight(label) for label in input_data]

    assert  len(traces) == len(input_data)

    # 遍历每一条曲线
    for j in range(len(traces)):

        # 如果输入的数据不是target，就跳过
        if input_data[j] != target_label:
            continue
        # 如果是，则进行下面的步骤
        count_of_used_traces +=1

        # 取出PoI
        a = [traces[j][PoIs[i]] for i in range(len(PoIs))]
        # Test each key

        for label in input_data_unique:

            # 协方差矩阵为nan，说明没有这个模板
            if np.isnan(covMatrix[label][0][0] ):
                print("label为"+ str(label)+ "的模板不存在")
                continue
            rv = multivariate_normal(meanMatrix[label], covMatrix[label],  allow_singular=True)
            p_kj = rv.pdf(a)

            # 这个输入有点问题，是1-254，没有0没有255
            P_k[label] += np.log(p_kj)

        # 去掉0，去掉255
        # 下标0-253 对应的key是1-254
        #print(P_k[1:255].argsort()[-5:])
        # print(P_k)

# 注意最右边的是排名第一的密钥
print(P_k[1:255].argsort()[-10:])
print("count_of_used_traces", count_of_used_traces)
print(P_k)

'''

#############################################

'''
# 下面是对一条曲线进行攻击

# 256种猜测的分数


P_k = np.zeros(num_of_class)


print("开始攻击")
target_part = 21
# 一个块中选择第几条
dijitiao = 0
# 一个part的曲线
traces = np.load(file_path + traces_file_name.format(target_part))


# 取出PoI
a = [traces[dijitiao][PoIs[i]] for i in range(len(PoIs))]
# 遍历每一个模板
for label in input_data_unique:
    if np.isnan(covMatrix[label][0][0]):
        print("label为"+ str(label)+ "的模板不存在")
        continue
    # Find p_{k,j}
    rv = multivariate_normal(meanMatrix[label], covMatrix[label],  allow_singular=True)
    p_kj = rv.pdf(a)
    # Add it to running total
    # 这个输入有点问题，是1-254，没有0没有255
    P_k[label] += np.log(p_kj)

# 去掉0，去掉255
# 下标0-253 对应的key是1-254
# 打印一下前10个
print(P_k[1:255].argsort()[-10:])
# print(P_k)
print(P_k)

'''

########################################################

# 下面是对多条曲线进行攻击,但是每个label只使用一条曲线进行攻击，计算所有攻击的成功率
P_k = np.zeros(num_of_class)    # num_of_class 种猜测的分数

# 攻击曲线使用多少个part
attack_part = 10
# 从第几个part开始
attack_part_start_index = 900

# 记录有多少条攻击成功了
success_num = 0
# 记录有多少条label为target_label的曲线被使用了
count_of_used_traces = 0

# 记录每一次攻击的成功率
success_rate_list=[]
print("开始攻击，对多条曲线, 求成功率")
for i in trange(attack_part):
    traces = np.load(file_path + traces_file_name.format(i + attack_part_start_index))  # 一个part的曲线
    # 降维
    if use_pca == True:
        traces = pca.transform(traces)
    input_data = np.loadtxt(file_path + input_data_file_name.format(i + attack_part_start_index), delimiter=',', dtype="int").tolist()
    input_data = [hammingWeight(label) for label in input_data]

    assert  traces.shape[0] == len(input_data)

    # 遍历每一条曲线
    for j in range(traces.shape[0]):

        count_of_used_traces += 1

        # 取出PoI
        a = [traces[j][PoIs[i]] for i in range(len(PoIs))]
        for label in input_data_unique:     # 测试每一个模板
            # 协方差矩阵为nan，说明没有这个模板
            if np.isnan(covMatrix[label][0][0] ):
                print("label为"+ str(label)+ "的模板不存在")
                continue
            rv = multivariate_normal(meanMatrix[label], covMatrix[label],  allow_singular=True)
            p_kj = rv.pdf(a)

            # 这个输入有点问题，是1-254，没有0没有255
            # 注意 这里不是+=了,不是求累计的分数
            P_k[label] = np.log(p_kj)   # 每一个模板的分数放进P_k
            # 运哥汉明重量的情况，原本输入数据应该是0-255，汉明重量应该是0-8，但是现在没有0没有255，汉明重量是1-7
            # 但是开的P_k大小是9,也就是[0,x,x,x,x,x,x,x,0]
            # 去掉第一个和最后一个
            # P_k = [x,x,x,x,x,x,x]
            # 下标0对应label为1 下标1对应label为2

        # 去头去尾
        P_k_temp = P_k[1:-1]
        # 如果最右边的数（猜测的密钥中分数最高的 所对应的下标）
        if P_k_temp.argsort()[-1] + 1 == input_data[j]:     # input_data[j] 是这条曲线对应的 label
            success_num += 1
        print("成功率",success_num / count_of_used_traces)
        success_rate_list.append(success_num / count_of_used_traces)
        # 下标0-253 对应的key是1-254



plt.plot(success_rate_list)
plt.show()
