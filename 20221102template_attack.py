import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.stats import multivariate_normal
from tqdm import trange
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# numPOIs: 有多少个PoI
# POIspacing: 至少相隔多元
def choosePoI(correlation_file_path, correlation_file_name, numPOIs, POIspacing):
    corr_trace = abs(np.load(correlation_file_path + correlation_file_name))[correlation_file_offset:]
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
# 归一化
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

print("loadtxt后用处理成汉明重量了！")
##########################
# 下面是路径参数 #
##########################

# file_path = r"F:/weixinzeng_wave_filter_trace_16bit/"
# file_path = r"F:/weixinzeng32sh8/"
# file_path = r"F:/another_CPU/5mhz_filter_8bit_new/"
# file_path = "F:/another_CPU/5mhz_filter_16bit/"
file_path = r"D:/ChipWhisperer5_52/cw/home/portable/chipwhisperer/tutorials/courses/sca101/example_aes/"

# file_path = r"D:/ChipWhisperer5_52/cw/home/portable/chipwhisperer/jupyter/courses/sca101/traces/"
# input_data_file_name = "aaadata{0}.txt"
input_data_file_name = "example_aes_inter_part{0}.txt"
# input_data_file_name = "lab3_3_zhongjianzhi_chaifen{0}.txt"
# traces_file_name = "arrPart{0}.npy"
traces_file_name = "example_aes_part{0}.npy"
# traces_file_name = "lab3_3_traces_chaifen{0}.npy"
# 注意要有相关性曲线的文件
# correlation_file_path = r"C:/Users/jfj/Desktop/PoI/another_CPU/5mhz_filter_16bit/"
correlation_file_path = r"C:/Users/jfj/Desktop/PoI/aes_test/"
correlation_file_name = "xiang_guan_xing.npy"
# correlation_file_name = "snr_data_better_var.npy"
correlation_file_offset = 5000  # 表示从第多少个点开始选PoI。因为第一个尖峰不看
correlation_file_offset = 0
num_of_each_part = 500  # 每个Part多少条曲线

numPOIs = 30    # 选择多少个PoI
POIspacing = 10     # PoI之间的间隔至少为多少

##########################
# 下面是建模参数 #
##########################
# 980
part = 150  # 选择多少个 part 进行建模
profiling_part_start_index = 0  # 建模曲线首先从哪一个part开始
num_of_class = 256  # 对于8bit HW是9类， 16bitHW是17类， 0-255有256类, 表示攻击的时候对多少类进行攻击
use_hammming = False
##########################
# 下面是PCA参数 #
##########################
use_pca = False  # 是否使用降维
pca_part = part  # 用多少 part 用来做 pca
pca_part_start_index = profiling_part_start_index    # 首先从哪一个part开始
pca_components = 3
# pca_method = ["full_traces", "mean_9", "mean_256", "mean_17"] 对应的下标
pca_method_select = 1

##########################

print("是否使用降维:", use_pca)
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


# 这个PoI对应的是 weixinzeng_wave_filter_trace_16bit （5hmz示波器）只看第二个尖峰，取10个点
# PoIs = [2445, 2455, 2466, 2476, 2487, 2497, 2508, 2518, 2529, 2540]

# 这个PoI对应的是 weixinzeng_wave_filter_trace_16bit （5hmz示波器）只看第二个尖峰，取15个点
# PoIs = [2445, 2451, 2458, 2465, 2472, 2478, 2485, 2492, 2499, 2506, 2512, 2519, 2526, 2533, 2540]

# 这个PoI对应的是 weixinzeng_wave_filter_trace_16bit （5hmz示波器）只看第二个尖峰，取20个点
# PoIs = [2445, 2448, 2451, 2454, 2458, 2461, 2464, 2467, 2471, 2474, 2477, 2481, 2484, 2487, 2490, 2494, 2497, 2500, 2503, 2507, 2510, 2513, 2517, 2520, 2523, 2526, 2530, 2533, 2536, 2540]

# 这个PoI对应的是 weixinzeng_wave_filter_trace_16bit （5hmz示波器）只看第二个尖峰，取25个点
# PoIs = [2445, 2448, 2452, 2456, 2460, 2464, 2468, 2472, 2476, 2480, 2484, 2488, 2492, 2496, 2500, 2504, 2508, 2512, 2516, 2520, 2524, 2528, 2532, 2536, 2540]

# 这个PoI对应的是 F:\another_CPU\5mhz_filter_8bit 看8个尖峰 , 每个尖峰取5个点
# PoIs = [5972, 5999, 6027, 6055, 6083,
# 6677, 6710, 6743, 6776, 6810,
# 7027, 7060, 7093, 7126, 7160,
# 7439, 7469, 7499, 7529, 7559,
# 7816, 7842, 7869, 7895, 7922,
# 8193, 8215, 8237, 8259, 8281,
# 8570, 8591, 8612, 8633, 8654,
# 8942, 8966, 8990, 9014, 9039]

# 这个PoI对应的是 F:\another_CPU\5mhz_filter_8bit 看8个尖峰 , 每个尖峰取10个点
# PoIs = [
# 5972, 5984, 5996, 6009, 6021, 6033, 6046, 6058, 6070, 6083,
# 6677, 6691, 6706, 6721, 6736, 6750, 6765, 6780, 6795, 6810,
# 7027, 7041, 7056, 7071, 7086, 7100, 7115, 7130, 7145, 7160,
# 7439, 7452, 7465, 7479, 7492, 7505, 7519, 7532, 7545, 7559,
# 7816, 7827, 7839, 7851, 7863, 7874, 7886, 7898, 7910, 7922,
# 8193, 8202, 8212, 8222, 8232, 8241, 8251, 8261, 8271, 8281,
# 8570, 8579, 8588, 8598, 8607, 8616, 8626, 8635, 8644, 8654,
# 8942, 8952, 8963, 8974, 8985, 8995, 9006, 9017, 9028, 9039]




# 这个PoI对应的是 F:\another_CPU\5mhz_filter_8bit 看8个尖峰 , 每个尖峰取15个点
# PoIs = [5935, 5950, 5965, 5980, 5995, 6010, 6025, 6040, 6055, 6070, 6085, 6100, 6115, 6130, 6146,
# 6676, 6688, 6701, 6714, 6727, 6740, 6753, 6766, 6779, 6792, 6805, 6818, 6831, 6844, 6857,
# 6986, 7003, 7020, 7037, 7054, 7072, 7089, 7106, 7123, 7140, 7158, 7175, 7192, 7209, 7227,
# 7438, 7449, 7461, 7473, 7484, 7496, 7508, 7520, 7531, 7543, 7555, 7566, 7578, 7590, 7602,
# 7792, 7804, 7817, 7830, 7843, 7856, 7869, 7882, 7894, 7907, 7920, 7933, 7946, 7959, 7972,
# 8192, 8202, 8213, 8224, 8235, 8245, 8256, 8267, 8278, 8289, 8299, 8310, 8321, 8332, 8343,
# 8558, 8569, 8580, 8592, 8603, 8615, 8626, 8638, 8649, 8660, 8672, 8683, 8695, 8706, 8718,
# 8920, 8934, 8948, 8962, 8976, 8990, 9004, 9019, 9033, 9047, 9061, 9075, 9089, 9103, 9118,
# ]

# 这个PoI对应的是 F:\another_CPU\5mhz_filter_8bit 看8个尖峰 , 每个尖峰取30个点
# PoIs = [
# 5972, 5975, 5979, 5983, 5987, 5991, 5994, 5998, 6002, 6006, 6010, 6014, 6017, 6021, 6025, 6029, 6033, 6037, 6040, 6044, 6048, 6052, 6056, 6060, 6063, 6067, 6071, 6075, 6079, 6083,
# 6677, 6681, 6686, 6690, 6695, 6699, 6704, 6709, 6713, 6718, 6722, 6727, 6732, 6736, 6741, 6745, 6750, 6754, 6759, 6764, 6768, 6773, 6777, 6782, 6787, 6791, 6796, 6800, 6805, 6810,
# 7027, 7031, 7036, 7040, 7045, 7049, 7054, 7059, 7063, 7068, 7072, 7077, 7082, 7086, 7091, 7095, 7100, 7104, 7109, 7114, 7118, 7123, 7127, 7132, 7137, 7141, 7146, 7150, 7155, 7160,
# 7439, 7443, 7447, 7451, 7455, 7459, 7463, 7467, 7472, 7476, 7480, 7484, 7488, 7492, 7496, 7501, 7505, 7509, 7513, 7517, 7521, 7525, 7530, 7534, 7538, 7542, 7546, 7550, 7554, 7559,
# 7816, 7819, 7823, 7826, 7830, 7834, 7837, 7841, 7845, 7848, 7852, 7856, 7859, 7863, 7867, 7870, 7874, 7878, 7881, 7885, 7889, 7892, 7896, 7900, 7903, 7907, 7911, 7914, 7918, 7922,
# 8193, 8196, 8199, 8202, 8205, 8208, 8211, 8214, 8217, 8220, 8223, 8226, 8229, 8232, 8235, 8238, 8241, 8244, 8247, 8250, 8253, 8256, 8259, 8262, 8265, 8268, 8271, 8274, 8277, 8281,
# 8570, 8572, 8575, 8578, 8581, 8584, 8587, 8590, 8593, 8596, 8598, 8601, 8604, 8607, 8610, 8613, 8616, 8619, 8622, 8625, 8627, 8630, 8633, 8636, 8639, 8642, 8645, 8648, 8651, 8654,
# 8942, 8945, 8948, 8952, 8955, 8958, 8962, 8965, 8968, 8972, 8975, 8978, 8982, 8985, 8988, 8992, 8995, 8998, 9002, 9005, 9008, 9012, 9015, 9018, 9022, 9025, 9028, 9032, 9035, 9039,
# ]

# 这个PoI对应的是 F:\another_CPU\5mhz_filter_8bit_new 看8个尖峰 , 每个尖峰取30个点
# PoIs = [5953, 5958, 5964, 5969, 5975, 5981, 5986, 5992, 5997, 6003, 6009, 6014, 6020, 6026, 6031, 6037, 6042, 6048, 6054, 6059, 6065, 6071, 6076, 6082, 6087, 6093, 6099, 6104, 6110, 6116,
# 6677, 6682, 6688, 6694, 6700, 6706, 6711, 6717, 6723, 6729, 6735, 6741, 6746, 6752, 6758, 6764, 6770, 6776, 6781, 6787, 6793, 6799, 6805, 6811, 6816, 6822, 6828, 6834, 6840, 6846,
# 7059, 7064, 7070, 7076, 7081, 7087, 7093, 7099, 7104, 7110, 7116, 7121, 7127, 7133, 7139, 7144, 7150, 7156, 7162, 7167, 7173, 7179, 7184, 7190, 7196, 7202, 7207, 7213, 7219, 7225,
# 7427, 7431, 7436, 7441, 7446, 7451, 7456, 7461, 7466, 7471, 7476, 7481, 7486, 7491, 7496, 7500, 7505, 7510, 7515, 7520, 7525, 7530, 7535, 7540, 7545, 7550, 7555, 7560, 7565, 7570,
# 7796, 7802, 7808, 7814, 7820, 7826, 7832, 7838, 7844, 7850, 7856, 7862, 7868, 7874, 7880, 7887, 7893, 7899, 7905, 7911, 7917, 7923, 7929, 7935, 7941, 7947, 7953, 7959, 7965, 7972,
# 8174, 8179, 8185, 8191, 8197, 8203, 8208, 8214, 8220, 8226, 8232, 8238, 8243, 8249, 8255, 8261, 8267, 8273, 8278, 8284, 8290, 8296, 8302, 8308, 8313, 8319, 8325, 8331, 8337, 8343,
# 8559, 8564, 8570, 8575, 8581, 8586, 8592, 8597, 8603, 8608, 8614, 8619, 8625, 8630, 8636, 8641, 8647, 8652, 8658, 8663, 8669, 8674, 8680, 8685, 8691, 8696, 8702, 8707, 8713, 8719,
# 8938, 8941, 8944, 8948, 8951, 8955, 8958, 8961, 8965, 8968, 8972, 8975, 8978, 8982, 8985, 8989, 8992, 8996, 8999, 9002, 9006, 9009, 9013, 9016, 9019, 9023, 9026, 9030, 9033, 9037]
# 这个PoI对应的是 F:\another_CPU\5mhz_filter_8bit 只看第二个尖峰,这个尖峰的每个点都取
# PoIs = [5957, 5958, 5959, 5960, 5961, 5962, 5963, 5964, 5965, 5966, 5967, 5968, 5969, 5970, 5971, 5972, 5973, 5974, 5975, 5976, 5977, 5978, 5979, 5980, 5981, 5982, 5983, 5984, 5985, 5986, 5987, 5988, 5989, 5990, 5991, 5992, 5993, 5994, 5995, 5996, 5997, 5998, 5999, 6000, 6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008, 6009, 6010, 6011, 6012, 6013, 6014, 6015, 6016, 6017, 6018, 6019, 6020, 6021, 6022, 6023, 6024, 6025, 6026, 6027, 6028, 6029, 6030, 6031, 6032, 6033, 6034, 6035, 6036, 6037, 6038, 6039, 6040, 6041, 6042, 6043, 6044, 6045, 6046, 6047, 6048, 6049, 6050, 6051, 6052, 6053, 6054, 6055, 6056, 6057, 6058, 6059, 6060, 6061, 6062, 6063, 6064, 6065, 6066, 6067, 6068, 6069, 6070, 6071, 6072, 6073, 6074, 6075, 6076, 6077, 6078, 6079, 6080, 6081, 6082, 6083, 6084, 6085, 6086, 6087, 6088, 6089, 6090, 6091, 6092, 6093, 6094, 6095, 6096, 6097, 6098, 6099, 6100, 6101, 6102, 6103, 6104, 6105, 6106, 6107, 6108, 6109, 6110, 6111, 6112, 6113, 6114]


# 这个PoI对应的是 AES D:\ChipWhisperer5_52\cw\home\portable\chipwhisperer\jupyter\courses\sca101\traces  lab3_3_yihuohoudezhiPart0.npy 的脚本自动挑选出来的
# PoIs = [1313, 1318, 1341, 2165, 1661]

# full 是 对全体 pca_part * num_of_each_part 条曲线做PCA_fit
# mean_9 是用汉明重量，对9条平均曲线做PCA_fit
# mean_256 是对256条平均曲线做PCA_fit

PoIs = choosePoI(correlation_file_path, correlation_file_name, numPOIs, POIspacing)

print("PoIs",PoIs)
scale= StandardScaler()
pca_method = ["full_traces", "mean_9", "mean_256", "mean_17"]

# PoI的个数
num_of_PoIs = len(PoIs)

if use_pca == True:
    pca = PCA(n_components=pca_components)
    print("开始降维")
    if pca_method[pca_method_select] == "full_traces":
        # 暂时存放用来pca的所有曲线
        temp_for_pca = np.zeros((pca_part * num_of_each_part, num_of_PoIs))
        # 遍历part
        for i in trange(pca_part):
            # 把用来PCA的曲线都放到   temp_for_pca 里面
            t = np.load(file_path + traces_file_name.format(i + pca_part_start_index))
            t = scale.fit_transform(t)  # 标准化
            t = t[:, PoIs]  # 挑出PoI
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
        temp_for_pca = np.zeros((pca_classes, num_of_PoIs))
        # 记录每一个label所对应的曲线数量
        pca_count = [0] * pca_classes
        for i in trange(pca_part):
            label_temp = np.loadtxt(file_path + input_data_file_name.format(i + pca_part_start_index), delimiter=' ',
                                    dtype="int")

            # 如果选择的是mean_9或mean_17就计算hamming重量
            if pca_method[pca_method_select] == "mean_9" or pca_method[pca_method_select] == "mean_17":
                label_temp = [hammingWeight(label) for label in label_temp]

            trace_temp = np.load(file_path + traces_file_name.format(i + pca_part_start_index))
            trace_temp = scale.fit_transform(trace_temp)
            trace_temp = trace_temp[:, PoIs]    # 挑出PoI

            # 遍历一个Part中所有曲线
            for index, label in enumerate(label_temp):
                pca_count[label] += 1
                # # 算出新的均值
                # meanMatrix[label] = meanMatrix[label] + (trace - meanMatrix[label]) / (count[label])  # 算出新的均值
                temp_for_pca[label] = temp_for_pca[label] + (trace_temp[index] - temp_for_pca[label]) / (
                pca_count[label])

        # 到这里计算出了均值，即temp_for_pca
        # print(temp_for_pca.shape)
        # plt.plot(temp_for_pca.T)
        # for i in range(temp_for_pca.shape[0]):
        #     plt.plot(temp_for_pca[i])
        plt.show()
        if temp_for_pca[0][0] == 0:
            print("均值PCA时剔除零值")
            print("temp_for_pca去掉第一行")
            temp_for_pca = temp_for_pca[1:]
        if temp_for_pca[-1][0] == 0:
            print("均值PCA时剔除零值")
            print("temp_for_pca去掉最后一行")
            temp_for_pca = temp_for_pca[:-1]

        # 拟合
        pca.fit(temp_for_pca)

if use_pca == False:
    # 初始化均值向量
    meanMatrix = [None] * num_of_class  # 对于每个类，存放每个PoI处的均值
    old_mean = [None] * num_of_class
    # 初始化协方差矩阵
    covMatrix = [None] * num_of_class
    # 初始化count ，用来记录每一个label下曲线的数目
    count = [None] * num_of_class
    for label in range(num_of_class):
        meanMatrix[label] = np.zeros(num_of_PoIs)
    for label in range(num_of_class):
        old_mean[label] = np.zeros(num_of_PoIs)
    for label in range(num_of_class):
        covMatrix[label] = np.zeros((num_of_PoIs, num_of_PoIs))
    for label in range(num_of_class):
        count[label] = 0

# 区别只有 num_of_class 变成pca_components
else:
    # 初始化均值向量
    meanMatrix = [None] * num_of_class  # 对于每个类，存放每个PoI处的均值
    old_mean = [None] * num_of_class
    # 初始化协方差矩阵
    covMatrix = [None] * num_of_class
    # 初始化count ，用来记录每一个label下曲线的数目
    count = [None] * num_of_class
    for label in range(num_of_class):
        meanMatrix[label] = np.zeros(pca_components)
    for label in range(num_of_class):
        old_mean[label] = np.zeros(pca_components)
    for label in range(num_of_class):
        covMatrix[label] = np.zeros((pca_components, pca_components))
    for label in range(num_of_class):
        count[label] = 0


print("开始建模")

for p in trange(part):
    input_data = np.loadtxt(file_path + input_data_file_name.format(p + profiling_part_start_index), delimiter=' ',
                            dtype="int")  # 读取一个part的输入数据
    if use_hammming:
        input_data = [hammingWeight(label) for label in input_data]  # 转化成汉明重量

    # 从profiling_part_start_index开始，读取这个 part 的曲线
    traces = np.load(file_path + traces_file_name.format(p + profiling_part_start_index))

    # traces = to_one2(traces)    # 标准化一下
    traces = scale.fit_transform(traces)
    traces = traces[:, PoIs]  # 只选取这些PoI。
    assert traces.shape[1] == len(PoIs)
    if use_pca == True:
        traces = pca.transform(traces)  # 使用PCA转化一下曲线,进一步减少特征
        assert traces.shape[1] == pca_components
    assert len(input_data) == traces.shape[0]


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
        # 在线计算协方差
        for i in range(traces.shape[1]):    # 这里不用num_of_PoIs，因为如果PCA了，特征数量更少，不是 num_of_PoIs
            for j in range(traces.shape[1]):
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
    for i in range(traces.shape[1]):
        for j in range(traces.shape[1]):
            covMatrix[label][i][j] = covMatrix[label][i][j] / (count[label] - 1)


# 求均值矩阵，协方差矩阵这部分验证了，没有问题

# 下面是对多条曲线进行攻击,但是每个label只使用一条曲线进行攻击，计算所有攻击的成功率
P_k = np.zeros(num_of_class)  # num_of_class 种猜测的分数
bayes_score = np.zeros(num_of_class)
##########################
# 下面是进行攻击部分的参数 #
##########################
use_bayes = True
attack_part = 3  # 攻击曲线使用多少个part

##########################
attack_part_start_index = part  # 从建模曲线的下一条曲线开始攻击

success_num = 0  # 记录有多少条攻击成功了
count_of_used_traces = 0  # 记录有多少条label为target_label的曲线被使用了
success_rate_list = []  # 记录每一次攻击的成功率
print("开始攻击，对多条曲线, 求成功率")
print("是否使用PCA",use_pca)
count_of_each_label_when_attacking = [0] * num_of_class
probability_of_each_label_when_attacking = [0] * num_of_class

for i in trange(attack_part):   # 先统计一下每个label的数目
    input_data = np.loadtxt(file_path + input_data_file_name.format(i + attack_part_start_index), delimiter=',',
                            dtype="int")
    if use_hammming:
        input_data = [hammingWeight(label) for label in input_data]

    for label in input_data:    # 遍历一个Part 中的所有输入
        count_of_each_label_when_attacking[label] += 1
test_total = 0
for label in range(num_of_class):   # 计算每一个label的概率
    probability_of_each_label_when_attacking[label] = ( count_of_each_label_when_attacking[label] / (attack_part * num_of_each_part) )
    test_total += probability_of_each_label_when_attacking[label]
print("test_total", test_total)
print("probability_of_each_label_when_attacking",probability_of_each_label_when_attacking)
for i in trange(attack_part):
    # 从 attack_part_start_index 开始，依次读取一个part的曲线
    traces = np.load(file_path + traces_file_name.format(i + attack_part_start_index))
    # traces = to_one2(traces)    # 标准化一下
    traces = scale.fit_transform(traces)
    traces = traces[:, PoIs]    # 只选取这些PoI。
    assert traces.shape[1] == len(PoIs)
    # 降维
    if use_pca:
        traces = pca.transform(traces)
        assert traces.shape[1] == pca_components
    input_data = np.loadtxt(file_path + input_data_file_name.format(i + attack_part_start_index), delimiter=',',
                            dtype="int")
    if use_hammming:
        input_data = [hammingWeight(label) for label in input_data]
    assert traces.shape[0] == len(input_data)
    for j in range(traces.shape[0]):    # 遍历这个part的每一条曲线
        count_of_used_traces += 1  # 使用的曲线数目 +1
        # a = [traces[j][PoIs[i]] for i in range(len(PoIs))]  # 取出PoI所对应的点，重组成a
        for label in range(num_of_class):  # 测试每一个模板 ，本来这里写的是 for label in input_data_unique
            # 有边界情况：协方差矩阵为nan，说明没有这个模板, 说明输入数据没有这个
            # 如果为0，说明输入数据只有1条曲线有这个，没法算出方差，方差为0
            if np.isnan(covMatrix[label][0][0]) or np.isnan(meanMatrix[label][0]) or covMatrix[label][0][0] == 0 or \
                    meanMatrix[label][0] == 0:
                print("label为" + str(label) + "的模板不存在,建模曲线数量不够")
                P_k[label] = float("-inf")  # 不存在的话，这个模板对应的分数就是负无穷
                continue
            rv = multivariate_normal(meanMatrix[label], covMatrix[label],allow_singular=True)  # 选择 label 所对应的模型
            p_kj = rv.pdf(traces[j])  # 将曲线带入公式
            # 注意 这里不是+=了,不是求累计的分数
            if use_bayes == False:
                P_k[label] = np.log(p_kj)  # 每一个模板的分数放进P_k
            else:
                P_k[label] = p_kj   # 取对数后会出现负无穷
        if use_bayes == False:   # 不使用Bayes
            # 如果最右边的数（猜测的密钥中分数最高的 所对应的下标） == label
            if P_k.argsort()[-1] == input_data[j]:  # input_data[j] 是这条曲线对应的 label，已经经过汉明重量的处理了
                success_num += 1
        else:   # 使用bayes
            denominator =  0
            for label in range(num_of_class):
                # P_k[label] 是 P(x|k)
                # probability_of_each_label_when_attacking[label]是 P(k)
                denominator += P_k[label] * probability_of_each_label_when_attacking[label]
            for label in range(num_of_class):   # 开始猜测
                bayes_score[label] = (probability_of_each_label_when_attacking[label] * P_k[label] ) / denominator
            if bayes_score.argsort()[-1] == input_data[j]:  # input_data[j] 是这条曲线对应的 label，已经经过汉明重量的处理了
                success_num += 1
        print("成功率", success_num / count_of_used_traces)
        success_rate_list.append(success_num / count_of_used_traces)

np.save("success_rate.npy",success_rate_list)
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