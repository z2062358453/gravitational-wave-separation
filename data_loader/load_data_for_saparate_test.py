import gc
import os
import pickle
# from generate_telescope_data import data_sample
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np

matplotlib.use('TkAgg')


class data_sample():
    def __init__(self, noiseE1, noiseE2, noiseE3, mass1_list, mass2_list, spin1z_list, spin2z_list, right_ascension_list,
                 declination_list, signal_E1_list, signal_E2_list, signal_E3_list, snr_E1_list, snr_E2_list, snr_E3_list):
        self.noiseE1 = noiseE1
        self.noiseE2 = noiseE2
        self.noiseE3 = noiseE3
        self.mass1_list = mass1_list
        self.mass2_list = mass2_list
        self.spin1z_list = spin1z_list
        self.spin2z_list = spin2z_list
        self.right_ascension_list = right_ascension_list
        self.declination_list = declination_list
        self.signal_E1_list = signal_E1_list
        self.signal_E2_list = signal_E2_list
        self.signal_E3_list = signal_E3_list
        self.snr_E1_list = snr_E1_list
        self.snr_E2_list = snr_E2_list
        self.snr_E3_list = snr_E3_list
    def print(self):
        print('mass1='+str(mass1_list))
        print('mass2=' + str(mass2_list))
        print('spin1z='+str(spin1z_list))
        print('spin2z='+str(spin2z_list))
        print('right_ascension='+str(right_ascension_list))
        print('declination='+str(declination_list))
    def help(self):
        print('noiseE1, noiseE2, noiseE3 are 32 s length noise of E1, E2 and E3')
        print('signal_E1_list, signal_E2_list and signal_E3_list are signals, and each have 10 samples')
        print('mass1_list, mass2_list are masses, and each have 10 masses')
        print('right_ascension_list and declination_list are the directions of the source of the signal')


# snr_range = (5,)

def file_iter(directory):
    file_list = os.listdir(directory)
    while True:
        for dir_item in file_list:
            yield os.path.join(directory,dir_item)

#
def sample_iter(datapath):
    data = [1,2,3]
    data_file_iter = file_iter(datapath)
    for file in data_file_iter:
        with open(file,'rb') as f:
            del data
            gc.collect()
            data = pickle.load(f)
        for sample in data:
            yield sample

def resize_to_snr(signal, pre_snr, after_snr):
    normal = after_snr/pre_snr
    return signal*normal

# qiefen signal
def cut_to_length(signal, time_length, sample_freq, peak_time):
    peak_now = np.argmax(signal)
    peak_after = int(peak_time*sample_freq)
    num = time_length*sample_freq
    cut_signal = np.zeros(num)
    if peak_now<peak_after:
        cut_signal[peak_after-peak_now:peak_after] = signal[0:peak_now]
    else:
        cut_signal[0:peak_after]=signal[peak_now-peak_after:peak_now]
    if num-peak_after>len(signal)-peak_now:
        cut_signal[peak_after:peak_after+len(signal)-peak_now-1] = signal[peak_now:-1]
    else:
        cut_signal[peak_after:-1]=signal[peak_now:peak_now+num-peak_after-1]
    return cut_signal

def random_cut_noise(noise, noise_rand_begin, noise_rand_end, freq, time):
    # begin = int(random.uniform(noise_rand_begin,noise_rand_end))
    begin = int(10*4096)
    end = int(begin+time*freq)
    return noise[begin:end]


def detach_test_peak_differ_example_cut_iter(SNRA, SNRB,datapath):

    data_iter_sample = sample_iter(datapath)
    noise_rand_begin = 4096
    noise_rand_end = int(63*4096-4*4096)
    A_index = 4
    B_index = 2

    example = next(data_iter_sample)
    signalA = example.signal_E1_list[A_index]
    MassA1 = example.mass1_list[A_index]
    MassA2 = example.mass2_list[A_index]
    print("MassA", MassA1, MassA2)
    signalB = example.signal_E1_list[B_index]
    SpinzA1 = example.spin1z_list[A_index]
    SpinzA2 = example.spin2z_list[A_index]
    print("spinzA", SpinzA1, SpinzA2)
    SpinzB1 = example.spin1z_list[B_index]
    SpinzB2 = example.spin2z_list[B_index]
    print("spinzB", SpinzB1, SpinzB2)
    MassB1 = example.mass1_list[B_index]
    MassB2 = example.mass2_list[B_index]
    print("MassB", MassB1, MassB2)
    SNR_A_PRE = example.snr_E1_list[A_index]
    print("归一化前SNRA:", SNR_A_PRE)
    SNR_B_PRE = example.snr_E1_list[B_index]
    print("归一化前SNRB:", SNR_B_PRE)
    # data = {
    #     "signalA": signalA,
    #     "SNR_A_PRE": SNR_A_PRE,
    #     "signalB": signalB,
    #     "SNR_B_PRE": SNR_B_PRE
    # }

    # 保存到 pkl 文件
    # with open("./Flask_project/signal_data.pkl", "wb") as f:
    #     pickle.dump(data, f)
    signal_A_reSNR = resize_to_snr(signalA,SNR_A_PRE,SNRA)
    # print("归一化后SNRA:",signal_A_reSNR)
    signal_B_reSNR = resize_to_snr(signalB, SNR_B_PRE, SNRB)
    # print("归一化后SNRB:",signal_B_reSNR)
    time_A = 3.7
    time_B_diff = [0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7]
    # time_B_diff = [-1.0]
    noise_E1 = random_cut_noise(example.noiseE1, noise_rand_begin,noise_rand_end,4096,4)

    signal_A_list = []
    signal_B_list = []
    noise_list = []
    signalA_cut = cut_to_length(signal_A_reSNR, 4, 4096, time_A)
    for td in time_B_diff:
        # noise_E1 = random_cut_noise(example.noiseE1, noise_rand_begin, noise_rand_end, 4096, 4)
        signal_A_list.append(signalA_cut)

        signal_B_cut = cut_to_length(signal_B_reSNR,4,4096,time_A+td)
        signal_B_list.append(signal_B_cut)
        noise_list.append(noise_E1)

    return signal_A_list, signal_B_list, noise_list
def detach_test_snr_example_differ_cut_iter(datapath):

    data_iter_sample = sample_iter(datapath)
    noise_rand_begin = 4096
    noise_rand_end = int(63*4096-4*4096)
    A_index = 4
    B_index = 2
    example = next(data_iter_sample)
    signalA = example.signal_E1_list[A_index]
    # print(signalA)
    # signalB = example.signal_E1_list[B_index]
    MassA1 = example.mass1_list[A_index]
    MassA2=example.mass2_list[A_index]
    SpinzA1=example.spin1z_list[A_index]
    SpinzA2=example.spin2z_list[A_index]
    print("spinzA",SpinzA1,SpinzA2)
    print("MassA",MassA1,MassA2)
    signalB = example.signal_E1_list[B_index]
    MassB1 = example.mass1_list[B_index]
    MassB2 = example.mass2_list[B_index]
    SpinzB1 = example.spin1z_list[B_index]
    SpinzB2 = example.spin2z_list[B_index]
    print("spinzB", SpinzB1, SpinzB2)
    print("MassB",MassB1,MassB2)
    SNR_A_PRE = example.snr_E1_list[A_index]
    print("归一化前SNRA:",SNR_A_PRE)
    SNR_B_PRE = example.snr_E1_list[B_index]
    print("归一化前SNRB:",SNR_B_PRE)
    noise_E1 = random_cut_noise(example.noiseE1, noise_rand_begin, noise_rand_end, 4096, 4)
    signal_A_list = []
    signal_B_list = []
    noise_list = []
    SNRA=10
    SNRB_range=[6, 8, 10, 12, 14, 16, 18, 20]
    for SNRB in SNRB_range:
            signal_A_reSNR = resize_to_snr(signalA,SNR_A_PRE,SNRA)
            # print("归一化后SNRA:",signal_A_reSNR)
            signal_B_reSNR = resize_to_snr(signalB, SNR_B_PRE, SNRB)
            # print("归一化后SNRB:",signal_B_reSNR)
            time_A = 3.7
            time_B = 3.5
            # noise_E1 = random_cut_noise(example.noiseE1, noise_rand_begin,noise_rand_end,4096,4)
            signalA_cut = cut_to_length(signal_A_reSNR, 4, 4096, time_A)
            signal_A_list.append(signalA_cut)
            signal_B_cut = cut_to_length(signal_B_reSNR, 4, 4096, time_B)
            signal_B_list.append(signal_B_cut)
            noise_list.append(noise_E1)

    return signal_A_list, signal_B_list, noise_list
def detach_test_peak_differ_cut_iter(SNRA, SNRB,datapath):

    data_iter_sample = sample_iter(datapath)
    noise_rand_begin = 4096
    noise_rand_end = int(63*4096-4*4096)
    time_A = 3.7
    time_B_diff=-1.0
    print("峰值时间差为：",time_B_diff)
    signal_A_list = []
    signal_B_list = []
    noise_list = []
    for example in data_iter_sample:
        # 调整信号A和信号B到目标SNR
        signalA = example.signal_E1_list[0]
        print(signalA)
        signalB = example.signal_E1_list[2]  # 使用 (i + 1) % len 确保索引不越界
        SNR_A_PRE = example.snr_E1_list[0]
        print(SNR_A_PRE)
        SNR_B_PRE = example.snr_E1_list[2]
        # SNRA=random.uniform(SNRA[0], SNRA[1])
        # SNRB = random.uniform(SNRB[0], SNRB[1])
        signal_A_reSNR = resize_to_snr(signalA, SNR_A_PRE, SNRA)
        signal_B_reSNR = resize_to_snr(signalB, SNR_B_PRE, SNRB)


        # 随机裁剪噪声
        # noise_E1 = random_cut_noise(example.noiseE1, noise_rand_begin, noise_rand_end, 4096, 4)

        # 裁剪信号A和信号B到指定长度，并在指定时间插入峰值
        signalA_cut = cut_to_length(signal_A_reSNR, 4, 4096, time_A)
        signalB_cut = cut_to_length(signal_B_reSNR, 4, 4096, time_A+time_B_diff)
        signal_A_list.append(signalA_cut)
        signal_B_list.append(signalB_cut)
        # noise_list.append(noise_E1)
        # if len(noise_list) >8000:
        #     break
        # print(len(noise_list))
        if len(signal_A_list) >= 1000:
            break

    return signal_A_list, signal_B_list

def detach_test_snr_differ_cut_iter(peak_time,datapath):
    # 获取数据迭代器
    data_iter_sample = sample_iter(datapath)

    # 定义噪声随机选择的起始和结束范围
    noise_rand_begin = 4096
    noise_rand_end = int(63 * 4096 - 4 * 4096)

    # 定义目标SNR值
    SNRA = 10
    SNRB = 20
    print("当前信噪比差值为：",SNRB-SNRA)
    SNRB_range = [6, 8, 10, 12, 14, 16, 18, 20]

    # 初始化信号和噪声的列表
    signal_A_list = []
    signal_B_list = []
    noise_list = []

    time_A = 3.7
    time_B_diff = -0.2
    # 遍历目标SNRB值
    for example in data_iter_sample:
            # 调整信号A和信号B到目标SNR
            signalA = example.signal_E1_list[0]
            print(signalA)
            signalB = example.signal_E1_list[2]  # 使用 (i + 1) % len 确保索引不越界
            SNR_A_PRE = example.snr_E1_list[0]
            print(SNR_A_PRE)
            SNR_B_PRE = example.snr_E1_list[2]
            signal_A_reSNR = resize_to_snr(signalA, SNR_A_PRE, SNRA)
            signal_B_reSNR = resize_to_snr(signalB, SNR_B_PRE, SNRB)

            # # 随机选择峰值时间
            # time_A = random.uniform(peak_time[0], peak_time[1]) * 4
            # print(time_A)
            # time_B = random.uniform(peak_time[0], peak_time[1]) * 4
            # print(time_B)

            # 随机裁剪噪声
            noise_E1 = random_cut_noise(example.noiseE1, noise_rand_begin, noise_rand_end, 4096, 4)

            # 裁剪信号A和信号B到指定长度，并在指定时间插入峰值
            signalA_cut = cut_to_length(signal_A_reSNR, 4, 4096, time_A)
            signalB_cut = cut_to_length(signal_B_reSNR, 4, 4096, time_A+time_B_diff)
            signal_A_list.append(signalA_cut)
            signal_B_list.append(signalB_cut)
            noise_list.append(noise_E1)
            # if len(noise_list) >8000:
            #     break
            print(len(noise_list))
            if len(noise_list)>=1000:
                break
    with open('noise_list.pkl','wb') as f:
            pickle.dump(noise_list,f)
    return signal_A_list, signal_B_list

def get_test_iter_for_peak_example(SNRA,SNRB,sample_length,freq,datapath):
        my_denoising_iter = detach_test_peak_differ_example_cut_iter(SNRA, SNRB,datapath)
        signal_A_list,signal_B_list,noise_list=my_denoising_iter
        # signal_A_list, signal_B_list, _ = my_denoising_iter
        print(len(signal_A_list))
        # noise_list = load_noise_list('noise_list.pkl')
        for signal_A,signal_B,noise in zip(signal_A_list,signal_B_list,noise_list):
            # noise=noise_list[0]
            # max_a=max(signal_A)
            # min_a=min(signal_A)
            # max_b=max(signal_B)
            # min_b=min(signal_B)
            # print("A信号最大峰值：",max_a)
            # print("A信号最小峰值：", min_a)
            # print("B信号最大峰值：", max_b)
            # print("B信号最大峰值：", min_b)
            batch_x0=noise + signal_A + signal_B
            batch_x = (noise + signal_A + signal_B) / np.max(noise + signal_A + signal_B)
            batch_y0 = signal_A
            batch_y1 = signal_B
            batch_y2 = signal_A  / np.max(signal_A)
            batch_y3 = signal_B  / np.max(signal_B)
            yield (batch_x0.reshape(-1, sample_length * freq, 1),
                   batch_y0.reshape(-1, sample_length * freq, 1),
                   batch_y1.reshape(-1, sample_length * freq, 1),
                   batch_x.reshape(-1, sample_length * freq, 1),
                   batch_y2.reshape(-1, sample_length * freq, 1),
                   batch_y3.reshape(-1, sample_length * freq, 1))
            del batch_x, batch_y1, batch_y2, signal_A, signal_B, noise
            gc.collect()


def get_test_iter_for_peak(SNRA, SNRB, sample_length, freq, datapath):
    my_denoising_iter = detach_test_peak_differ_cut_iter(SNRA, SNRB, datapath)
    signal_A_list, signal_B_list = my_denoising_iter
    # print(len(signal_A_list))
    noise_list = load_noise_list('noise_list.pkl')
    for signal_A, signal_B, noise in zip(signal_A_list, signal_B_list, noise_list):
        batch_x = (noise + signal_A + signal_B) / np.max(noise + signal_A + signal_B)
        batch_y1 = signal_A / np.max(signal_A)
        batch_y2 = signal_B / np.max(signal_B)
        yield (batch_x.reshape(-1, sample_length * freq, 1), batch_y1.reshape(-1, sample_length * freq, 1),
               batch_y2.reshape(-1, sample_length * freq, 1))
        del batch_x, batch_y1, batch_y2, signal_A, signal_B, noise
        gc.collect()

def get_test_iter_for_snr_example(sample_length, freq, datapath):
        my_denoising_iter = detach_test_snr_example_differ_cut_iter(datapath)
        # my_denoising_iter = detach_test_peak_differ_example_cut_iter(SNRA, SNRB,datapath)
        signal_A_list,signal_B_list,noise_list=my_denoising_iter
        print(len(signal_A_list))

        for signal_A, signal_B, noise in zip(signal_A_list, signal_B_list, noise_list):
            # max_a = max(signal_A)
            # min_a = min(signal_A)
            # max_b = max(signal_B)
            # min_b = min(signal_B)
            # print("A信号最大峰值：", max_a)
            # print("A信号最小峰值：", min_a)
            # print("B信号最大峰值：", max_b)
            # print("B信号最大峰值：", min_b)
            batch_x0 = noise + signal_A + signal_B
            batch_x = (noise + signal_B+signal_A) / np.max(noise + signal_B+signal_A)
            batch_y0 = signal_A
            batch_y1 = signal_B
            batch_y2 = signal_A / np.max(signal_A)
            batch_y3 = signal_B / np.max(signal_B)
            yield (batch_x0.reshape(-1, sample_length * freq, 1),
                   batch_y0.reshape(-1, sample_length * freq, 1),
                   batch_y1.reshape(-1, sample_length * freq, 1),
                   batch_x.reshape(-1, sample_length * freq, 1),
                   batch_y2.reshape(-1, sample_length * freq, 1),
                   batch_y3.reshape(-1, sample_length * freq, 1))
            del batch_x, batch_y1, batch_y2, signal_A, signal_B, noise
            gc.collect()

def load_noise_list(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# 调用函数加载 noise_list
# noise_list = load_noise_list('noise_list.pkl')
def get_test_iter_for_snr(peak_time, sample_length, freq, datapath):
    my_denoising_iter = detach_test_snr_differ_cut_iter(peak_time, datapath)
    # my_denoising_iter = detach_test_peak_differ_example_cut_iter(SNRA, SNRB,datapath)
    signal_A_list, signal_B_list = my_denoising_iter
    print(len(signal_A_list))
    noise_list = load_noise_list('noise_list.pkl')
    for signal_A, signal_B, noise in zip(signal_A_list, signal_B_list, noise_list):
        # batch_x = (noise+signal_B) / np.max(noise+signal_B)
        batch_x1 = (noise + signal_B+signal_A) / np.max(noise + signal_B+signal_A)
        batch_y1 = signal_A / np.max(signal_A)
        batch_y2 = signal_B / np.max(signal_B)
        yield (batch_x1.reshape(-1, sample_length * freq, 1),
               # batch_x1.reshape(-1, sample_length * freq, 1),
               batch_y1.reshape(-1, sample_length * freq, 1),
               batch_y2.reshape(-1, sample_length * freq, 1))
        del batch_x1, batch_y1, batch_y2, signal_A, signal_B, noise
        gc.collect()

def get_test_iter_for_snr_peak_time_example(SNRA,SNRB,sample_length, freq, datapath):
        my_denoising_iter2 = detach_test_snr_example_differ_cut_iter(datapath)
        my_denoising_iter1 = detach_test_peak_differ_example_cut_iter(SNRA, SNRB,datapath)
        signal_A_list, signal_B_list, noise_list = my_denoising_iter1
        signal_A_list1, signal_B_list1, noise_list1 = my_denoising_iter2
        print(len(signal_A_list))

        for signal_A_peak, signal_B_peak, noise_peak,signal_A_snr, signal_B_snr, noise_snr in zip(signal_A_list, signal_B_list, noise_list,signal_A_list1, signal_B_list1, noise_list1):
            # max_a = max(signal_A)
            # min_a = min(signal_A)
            # max_b = max(signal_B)
            # min_b = min(signal_B)
            # print("A信号最大峰值：", max_a)
            # print("A信号最小峰值：", min_a)
            # print("B信号最大峰值：", max_b)
            # print("B信号最大峰值：", min_b)
            # max_a=max( batch_peak_y)
            # min_a = min( batch_peak_y)
            # print(max_a)
            # print(min_a)
            batch_peak_y = signal_A_peak + signal_B_peak
            batch_snr_y = signal_A_snr + signal_B_snr
            yield (batch_peak_y.reshape(-1, sample_length * freq, 1),
                   batch_snr_y.reshape(-1, sample_length * freq, 1))
            del batch_peak_y,batch_snr_y
            gc.collect()
# test_iter=get_test_iter_for_peak_example(SNRA=10,
#                                          SNRB=10,
#                                          sample_length=4,
#                                          freq=4096,
#                                          datapath='H://test')
# n=0
# for x0,y0,y1,x,y2,y3 in  test_iter:
#     n+=1
#     print(n)
# signal_A_list, signal_B_list, noise_list = detach_test_peak_differ_cut_iter(10,10,datapath='H://test')
# for signal_A,signal_B,noise in zip(signal_A_list, signal_B_list, noise_list):
#     plt.plot(signal_A)
#     plt.plot(signal_B)
#     plt.plot(noise)
#     plt.show()
# signal_A_list, signal_B_list, noise_list = detach_test_peak_differ_example_cut_iter(12,12,datapath='E://test')
# for signal_A,signal_B,noise in zip(signal_A_list, signal_B_list, noise_list):
#     plt.plot(signal_A)
#     plt.plot(signal_B)
#     plt.plot(noise)
#     plt.show()
# test_datapath = 'H://test'
# sample_length = 4
#         # snr_list=[50,45,40,35]
# peak_range = (0.5, 0.95)
# freq = 4096
# test_data1 = get_test_iter_for_peak_example(SNRA=12,
#                                                    SNRB=12,
#                                                    sample_length=sample_length,
#                                                    freq=freq,
#                                                    datapath=test_datapath)

# signal_A_list, signal_B_list, noise_list = detach_test_snr_example_differ_cut_iter(peak_time)
# for signal_A,signal_B,noise in zip(signal_A_list, signal_B_list, noise_list):
#     plt.plot(signal_A)
#     plt.plot(signal_B)
#     plt.show()
# n=0
# signal_A_list, signal_B_list, noise_list = detach_test_snr_differ_cut_iter(peak_time, datapath='H://test')
# for signal_A,signal_B,noise in zip(signal_A_list, signal_B_list, noise_list):
#     n+=1
#     print(n)
def pad_first_to_length(signal_E1, signal_E2, signal_E3, length, freq, peak_range):
    signal_peak_1 = np.argmax(signal_E1)
    signal_peak_2 = np.argmax(signal_E2)
    signal_peak_3 = np.argmax(signal_E3)
    signal_after_peak_1 = np.size(signal_E1) - np.argmax(signal_E1)
    signal_after_peak_2 = np.size(signal_E2) - np.argmax(signal_E2)
    signal_after_peak_3 = np.size(signal_E3) - np.argmax(signal_E3)
    delta_2 = signal_peak_2 - signal_peak_1
    delta_3 = signal_peak_3 - signal_peak_1
    signal_E1_padding = np.zeros(int(length*freq))
    signal_E2_padding = np.zeros(int(length * freq))
    signal_E3_padding = np.zeros(int(length * freq))
    peak_norm = random.uniform(peak_range[0],peak_range[1])
    peak_1 = int(peak_norm*length*freq)
    peak_2 = peak_1+delta_2
    peak_3 = peak_1+delta_3
    after_peak_1 = int(length*freq)-peak_1
    after_peak_2 = int(length*freq)-peak_2
    after_peak_3 = int(length*freq)-peak_3

    if (peak_1 < signal_peak_1):
        signal_E1_padding[:peak_1] = signal_E1[signal_peak_1 - peak_1:signal_peak_1]
    else:
        signal_E1_padding[peak_1 - signal_peak_1:peak_1] = signal_E1[0:signal_peak_1]
    if (signal_after_peak_1 < after_peak_1):
        signal_E1_padding[peak_1:peak_1 + signal_after_peak_1] = signal_E1[signal_peak_1:]
    else:
        signal_E1_padding[peak_1:] = signal_E1[signal_peak_1:signal_peak_1 + after_peak_1]

    if (peak_2 < signal_peak_2):
        signal_E2_padding[:peak_2] = signal_E2[signal_peak_2 - peak_2:signal_peak_2]
    else:
        signal_E2_padding[peak_2 - signal_peak_2:peak_2] = signal_E2[0:signal_peak_2]
    if (signal_after_peak_2 < after_peak_2):
        signal_E2_padding[peak_2:peak_2 + signal_after_peak_2] = signal_E2[signal_peak_2:]
    else:
        signal_E2_padding[peak_2:] = signal_E2[signal_peak_2:signal_peak_2 + after_peak_2]

    if (peak_3 < signal_peak_3):
        signal_E3_padding[:peak_3] = signal_E3[signal_peak_3 - peak_3:signal_peak_3]
    else:
        signal_E3_padding[peak_3 - signal_peak_3:peak_3] = signal_E3[0:signal_peak_3]
    if (signal_after_peak_3 < after_peak_3):
        signal_E3_padding[peak_3:peak_3 + signal_after_peak_3] = signal_E3[signal_peak_3:]
    else:
        signal_E3_padding[peak_3:] = signal_E3[signal_peak_3:signal_peak_3 + after_peak_3]

    return signal_E1_padding, signal_E2_padding, signal_E3_padding



def detach_sample_cut_iter(sample_length,snr_range, peak_range,freq):
    data_iter_sample = sample_iter()
    noise_rand_begin = 2048
    noise_rand_end = int(63*2048-sample_length*freq)
    all_num = sample_length*freq
    for data_sample_ in data_iter_sample:
        iter_index = 0
        for signal_E1, signal_E2, signal_E3, snr_E1, snr_E2, snr_E3 \
                in zip(data_sample_.signal_E1_list,data_sample_.signal_E2_list,data_sample_.signal_E3_list,data_sample_.snr_E1_list,\
                       data_sample_.snr_E2_list, data_sample_.snr_E3_list):
            second_rand = random.randint(1,9)
            second_index = (iter_index+second_rand)%10
            second_snr = random.uniform(snr_range[0],snr_range[1])
            second_snr_norm = second_snr/np.sqrt(data_sample_.snr_E1_list[second_index]**2+data_sample_.snr_E2_list[second_index]**2 \
                                          +data_sample_.snr_E3_list[second_index]**2)


            snr = random.uniform(snr_range[0],snr_range[1])
            snr_norm = snr/np.sqrt(snr_E1**2+snr_E2**2+snr_E3**2)
            noise_begin_E1 = random.randint(noise_rand_begin,noise_rand_end)
            noise_E1 = data_sample_.noiseE1[noise_begin_E1:noise_begin_E1+all_num]
            noise_begin_E2 = random.randint(noise_rand_begin, noise_rand_end)
            noise_E2 = data_sample_.noiseE1[noise_begin_E2:noise_begin_E2 + all_num]
            noise_begin_E3 = random.randint(noise_rand_begin, noise_rand_end)
            noise_E3 = data_sample_.noiseE1[noise_begin_E3:noise_begin_E3 + all_num]
            iter_index = (iter_index+1)%10
            signal_E1_1 = snr_norm*signal_E1
            signal_E1_2 = second_snr_norm*data_sample_.signal_E1_list[second_index]
            signal_E2_1 = snr_norm*signal_E2
            signal_E2_2 = second_snr_norm*data_sample_.signal_E2_list[second_index]
            signal_E3_1 = snr_norm*signal_E3
            signal_E3_2 = second_snr_norm*data_sample_.signal_E3_list[second_index]

            signal_E1_1_padding, signal_E2_1_padding, signal_E3_1_padding = pad_first_to_length(signal_E1_1,
                                                                                                signal_E2_1,
                                                                                                signal_E3_1,
                                                                                                sample_length,
                                                                                                freq,
                                                                                                peak_range)
            signal_E1_2_padding, signal_E2_2_padding, signal_E3_2_padding = pad_first_to_length(signal_E1_2,
                                                                                                signal_E2_2,
                                                                                                signal_E3_2,
                                                                                                sample_length, freq,
                                                                                                peak_range)

            yield noise_E1, signal_E1_1_padding, signal_E1_2_padding, \
                noise_E2, signal_E2_1_padding, signal_E2_2_padding, \
                noise_E3, signal_E3_1_padding, signal_E3_2_padding


#
# my_iter = detach_sample_cut_iter(4,(8,30),(0.5,0.95),2048)
#
# for noise_1, signal_1_1, signal_1_2, noise_2, signal_2_1, signal_2_2, noise_3, signal_3_1, signal_3_2 in my_iter:
#     plt.plot((noise_1+signal_1_1+signal_1_2)*1e23)
#     plt.plot(signal_1_1/np.max(signal_1_1))
#     plt.plot(signal_1_2/np.max(signal_1_2))
#
#
#
#     plt.show()