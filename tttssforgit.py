# 从0开始数，偶数曲线，随机，奇数曲线，固定
# 采集曲线分块，防止内存溢出
# 采集完整的AES，使用自带的板子！！！！
import numpy as np
import random
from tqdm import tnrange
from tqdm.auto import tqdm
from numpy import savetxt
import galois
import gc

################下方这里是参数部分，可以修改#######
traces_per_part = 500  # 每一块文件多少条曲线
part = 200  # 分成多少块文件
part_start_index = 0  # 序号从多少号开始。如果是初始采集，就设置为0
scope.adc.samples = 20000  # 总共需要采集多少个点

# scope.adc.decimate = 4  # 可以调节下采样比例
# 曲线文件的文件名前缀
filename_of_traces = "aes_capture"
##############################################

global_counter = 0  # 全局的counter，用来判断这条曲线是奇数还是偶数
total_traces = part * traces_per_part  # 总共要采集的曲线数目
pbar = tqdm(total=total_traces)


# 这个函数用于采集并保存一份文件
def mainProcess(num_of_traces):
    traces_array = []
    text_in_array = []
    text_out_array = []

    for i in tnrange(0, num_of_traces):

        reset_target(scope)  # 对板子复位
        global quanjucounter
        global pbar

        if quanjucounter % 2 == 0:  # # 如果是偶数曲线，按照全随机的方案
            list_to_send = [1, 2, 3]  # 表示输入的数据
            msg = bytearray(list_to_send)
            scope.arm()
            target.simpleserial_write('g', msg)  # 这里的'g'命令要和c文件中的对应一致
            time.sleep(0.05)  # ❗这里必须要sleep,如果采集出错，尝试对睡眠时间进行调整

        # 如果是奇数曲线，按照固定的方案
        else:
            list_to_send = [4, 5, 6]
            msg = bytearray(list_to_send)
            scope.arm()
            target.simpleserial_write('g', msg)
            # 注意如果要多次发送命令，则发送后要睡一下
            time.sleep(0.05)
        ret = scope.capture()
        if ret:
            print("Target timed out!")
        trace = scope.get_last_trace()  # 获得曲线
        recv_msg = target.simpleserial_read('r', 1)  # 这里的r命令要和c文件中对应一致

        traces_array.append(list(trace))
        text_in_array.append(list_to_send)
        text_out_array.append(list(recv_msg))
        quanjucounter += 1
        pbar.update(1)  # 更新进度条

    return np.array(traces_array), np.array(text_in_array), np.array(text_out_array)


for p in range(part):  # 遍历所有的part
    traces_arr, text_in_arr, text_out_arr = mainProcess(traces_per_part)
    np.save(filename_of_traces + "tracesPart{0}.npy".format(p + part_start_index), traces_arr)
    np.save(filename_of_traces + "textinPart{0}.npy".format(p + part_start_index), text_in_arr)
    np.save(filename_of_traces + "textoutPart{0}.npy".format(p + part_start_index), text_out_arr)
    counter += 1
    # del traces_array  # 回收
    # gc.collect()

# todo:发现有时候会采集错误，就是一个曲线是一样的数值
# 并且好几条曲线都是一样的数值
# 应该写几行代码做个错误判断
# [[ 32512  32512  32512 ...  32512  32512  32512]
# [ 32512  32512  32512 ...  32512  32512  32512]
# [ 32512  32512  32512 ...  32512  32512  32512]]
# 一般发生在开始的几条曲线上
pbar.close()
