import numpy as np
def better_var(X_list):
    count = 0
    mean = 0
    M2 = 0
    for i in range(len(X_list)):
        count += 1
        delta = X_list[i] - mean
        mean += delta / count
        delta2 = X_list[i] - mean
        M2 += delta * delta2
    print(M2 / (count))
def online_cov(X_list,Y_list):
    meanx = meany = C = n = 0

    for i in range(len(X_list)):
        x = X_list[i]
        y = Y_list[i]
        n += 1
        dx = x - meanx

        # new meanx
        meanx += dx / n

        # new meany
        meany += (y - meany) / n
        #C =( C * (n-1) +  dx * (y - meany) ) / n
        C += dx * (y - meany)

    print(C / (n))
X = np.random.rand(10000)
Y = np.random.rand(10000)
#X = [1,2,4,6,8,5,6][:6]
#Y = [3,6,2,6,9,7,8][:6]
#online_cov(X, Y)
better_var(X)
print("numpy_result")

#print(np.cov(X, Y)[0][1])
print(np.var(X))


# 用numpy测一下数据
def test_use_naive_method():
    PoIs = [3134, 4140, 5160, 6170, 7190, 8190, 9220, 10220, 11240, 12270, 13300, 14280, 15310, 16330, 17330, 18350,
            19360, 20430, 21390, 23400, 25430, 27450, 29470]

    part = 10
    num_of_each_part = 500
    input_data = []
    for i in range(part):
        # 收集一下所有的输入数据
        input_data += np.loadtxt(r"F:/tracexinzeng32sh8/aaadata{0}.txt".format(i), delimiter=',', dtype="int").tolist()

    # 去重一下
    input_data_unique = np.unique(input_data)
    alltraces = [None] *256
    alltraces = np.zeros((part * num_of_each_part, len(PoIs)))
    for i in range(part):
        # 读取一个part的输入数据
        input_data = np.loadtxt("F:/tracexinzeng32sh8/aaadata{0}.txt".format(i), delimiter=' ', dtype="int")

        # 一个part的曲线
        traces = np.load("F:/tracexinzeng32sh8/arrPart{0}.npy".format(i))

        # 放入
        alltraces[i * num_of_each_part:i * num_of_each_part + 500] = traces

        M = np.means(alltraces,axis = 0)
        print("均值")
        print(M)

        COV = np.cov