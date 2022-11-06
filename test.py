import numpy as np

# 43
key = 0x2b
traces = np.load(r"D:/ChipWhisperer5_52/cw/home/portable/chipwhisperer/jupyter/courses/sca101/traces/lab3_3_traces0.npy")
input = np.load(r"D:/ChipWhisperer5_52/cw/home/portable/chipwhisperer/jupyter/courses/sca101/traces/lab3_3_textin.npy")

print(input)



for i in range(len(input)):
    input[i] = input[i] ^ key

input = input[:,0]
np.save(r"D:/ChipWhisperer5_52/cw/home/portable/chipwhisperer/jupyter/courses/sca101/traces/lab3_3_yihuohoudezhiPart0.npy",input)

print(input)


n = np.load(r"D:/ChipWhisperer5_52/cw/home/portable/chipwhisperer/jupyter/courses/sca101/traces/lab3_3_yihuohoudezhiPart0.npy")

print(n)



