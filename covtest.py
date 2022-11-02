import numpy as np

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
online_cov(X, Y)

print("numpy_result")

print(np.cov(X, Y)[0][1])

