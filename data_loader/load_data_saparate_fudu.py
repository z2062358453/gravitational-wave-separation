import gc
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
# from generate_telescope_data import data_sample
# matplotlib.use('TkAgg')

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
        print('noiseE1, noiseE2, noiseE3 are 128 s length noise of E1, E2 and E3')
        print('signal_E1_list, signal_E2_list and signal_E3_list are signals, and each have 1 samples')
        print('mass1_list, mass2_list are masses, and each have 1 masses')
        print('right_ascension_list and declination_list are the directions of the source of the signal')


def file_iter(directory):
    file_list = os.listdir(directory)
    # while True:
    for dir_item in file_list:
        yield os.path.join(directory,dir_item)


def sample_iter(datapath):
    data = [1,2,3]
    # data_file_iter = list(file_iter(datapath))
    data_file_iter = file_iter(datapath)
    # while True:
        # random.shuffle(data_file_iter)
        # print(data_file_iter)

    for file in data_file_iter:
            # print(file)
            with open(file,'rb') as f:
                del data
                gc.collect()
                data = pickle.load(f)
            for sample in data:
                yield sample

# 将三个信号填充到指定长度的函数
def pad_first_to_length(signal_E1, signal_E2, signal_E3, length, freq, peak_range):
    # 找到信号 signal_E1 的最大值所在的索引，即信号的峰值位置
    signal_peak_1 = np.argmax(signal_E1)
    signal_peak_2 = np.argmax(signal_E2)
    signal_peak_3 = np.argmax(signal_E3)
    #  计算信号 signal_E1 峰值位置之后的长度。
    signal_after_peak_1 = np.size(signal_E1) - np.argmax(signal_E1)
    signal_after_peak_2 = np.size(signal_E2) - np.argmax(signal_E2)
    signal_after_peak_3 = np.size(signal_E3) - np.argmax(signal_E3)
    # 计算 signal_E2 和 signal_E3的峰值位置相对于 signal_E1 的峰值位置的偏移量。
    delta_2 = signal_peak_2 - signal_peak_1
    delta_3 = signal_peak_3 - signal_peak_1
    # 初始化长度为 length*freq 的三个零数组 signal_E1_padding、signal_E2_padding 和 signal_E3_padding，用于存储填充后的信号。
    signal_E1_padding = np.zeros(int(length*freq))
    signal_E2_padding = np.zeros(int(length * freq))
    signal_E3_padding = np.zeros(int(length * freq))
    # 生成随机峰值比例 peak_norm
    peak_norm = random.uniform(peak_range[0],peak_range[1])
    # 根据该比例计算三个信号的峰值位置
    peak_1 = int(peak_norm*length*freq)
    peak_2 = peak_1+delta_2
    peak_3 = peak_1+delta_3

    # 计算三个信号相对于指定长度的偏移量和剩余长度。
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


# snr_low_list = [30, 26, 22, 17, 12, 12, 17, 22, 26]
# snr_high_list = [26, 22, 17, 12, 8, 8, 12, 17, 22]
def detach_sample_cut_iter(sample_length,snr_low_list,snr_high_list, peak_range,freq,snr_change_time,datapath):
    data_iter_sample = sample_iter(datapath)
    # noise_rand_begin = 2048
    noise_rand_begin = 4096
    noise_rand_end = int(63*4096-sample_length*freq)
    all_num = sample_length*freq
    gen_num = 0
    obj_snr_index = 0
    for data_sample_ in data_iter_sample:
        iter_index = 0
        for signal_E1, signal_E2, signal_E3, snr_E1, snr_E2, snr_E3 \
                in zip(data_sample_.signal_E1_list,data_sample_.signal_E2_list,data_sample_.signal_E3_list,data_sample_.snr_E1_list,\
                       data_sample_.snr_E2_list, data_sample_.snr_E3_list):
            # print(signal_E1.shape)
            second_rand = random.randint(1,9)
            second_index = (iter_index+second_rand)%10
            snr_low = snr_low_list[obj_snr_index]
            snr_high = snr_high_list[obj_snr_index]
            # second_snr = random.uniform(snr_range[0],snr_range[1])
            if gen_num > 0 and gen_num % snr_change_time == 0:
                obj_snr_index = (obj_snr_index + 1) % len(snr_low_list)
                # print('next snr is:', str(snr_list[obj_snr_index]))
            second_snr = random.uniform(snr_low, snr_high)
            snr = random.uniform(snr_low, snr_high)
            second_snr_norm = second_snr/np.sqrt(data_sample_.snr_E1_list[second_index]**2+data_sample_.snr_E2_list[second_index]**2 \
                                          +data_sample_.snr_E3_list[second_index]**2)


            # snr = random.uniform(snr_range[0],snr_range[1])
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
            gen_num = gen_num + 1
            yield noise_E1, signal_E1_1_padding, signal_E1_2_padding,\
                noise_E2, signal_E2_1_padding, signal_E2_2_padding, \
                noise_E3, signal_E3_1_padding, signal_E3_2_padding


def get_train_batch_iter(batch_size,sample_length,snr_low_list,snr_high_list, peak_range,freq, snr_change_time, datapath):
    my_denoising_iter = detach_sample_cut_iter(sample_length,snr_low_list,snr_high_list, peak_range,freq,snr_change_time,datapath)
    count = 1
    batch_x = []
    batch_y = []
    for noise_E1, signal_E1_1, signal_E1_2, noise_E2, signal_E2_1, signal_E2_2, noise_E3, signal_E3_1, signal_E3_2 in my_denoising_iter:
        # print(noise_E1.shape)
        # print(strain.shape)
        if count == 1:
            batch_x = (noise_E1 + signal_E1_1 + signal_E1_2)*1e23
            batch_y = (signal_E1_1 + signal_E1_2)*1e23
        else:
            batch_x = np.concatenate(
                (batch_x, (noise_E1 + signal_E1_1 + signal_E1_2)*1e23))
            batch_x = np.concatenate(
                (batch_x, (noise_E2 + signal_E2_1 + signal_E2_2)*1e23))
            batch_x = np.concatenate(
                (batch_x, (noise_E3 + signal_E3_1 + signal_E3_2)*1e23))
            batch_y = np.concatenate((batch_y, (signal_E1_1 + signal_E1_2)*1e23))
            batch_y = np.concatenate((batch_y, (signal_E2_1 + signal_E2_2)*1e23))
            batch_y = np.concatenate((batch_y, (signal_E3_1 + signal_E3_2)*1e23))
        if count == batch_size:
            yield (batch_x.reshape(-1, sample_length * freq, 1), batch_y.reshape(-1, sample_length * freq, 1))
            del batch_x, batch_y, noise_E1, signal_E1_1, signal_E1_2, noise_E2, signal_E2_1, signal_E2_2, noise_E3, signal_E3_1, signal_E3_2
            gc.collect()
        count = count + 3
        if count > batch_size:
            count = 1
def get_train_batch_iter_saparate(batch_size,sample_length,snr_low_list,snr_high_list, peak_range,freq, snr_change_time, datapath):
    my_denoising_iter = detach_sample_cut_iter(sample_length, snr_low_list, snr_high_list, peak_range, freq,
                                               snr_change_time, datapath)
    count = 1
    batch_x = []
    batch_y_1 = []
    batch_y_2 = []
    for noise_E1, signal_E1_1, signal_E1_2, noise_E2, signal_E2_1, signal_E2_2, noise_E3, signal_E3_1, signal_E3_2 in my_denoising_iter:
        # print(noise_E1.shape)
        # print(strain.shape)
        if count == 1:
            batch_x = (noise_E1 + signal_E1_1 + signal_E1_2)*1e23
            batch_y_1 = signal_E1_1*1e23
            batch_y_2 = signal_E1_2*1e23
        else:
            batch_x = np.concatenate(
                (batch_x, (noise_E1 + signal_E1_1 + signal_E1_2)*1e23))
            batch_x = np.concatenate(
                (batch_x, (noise_E2 + signal_E2_1 + signal_E2_2)*1e23))
            batch_x = np.concatenate(
                (batch_x, (noise_E3 + signal_E3_1 + signal_E3_2)*1e23))
            batch_y_1 = np.concatenate((batch_y_1, signal_E1_1*1e23))
            batch_y_2 = np.concatenate((batch_y_2, signal_E1_2*1e23))
            batch_y_1 = np.concatenate((batch_y_1, signal_E2_1*1e23))
            batch_y_2 = np.concatenate((batch_y_2, signal_E2_2*1e23))
            batch_y_1 = np.concatenate((batch_y_1, signal_E3_1*1e23))
            batch_y_2 = np.concatenate((batch_y_2, signal_E3_2*1e23))
        if count == batch_size:
            yield (batch_x.reshape(-1, sample_length * freq, 1), batch_y_1.reshape(-1, sample_length * freq, 1),
                   batch_y_2.reshape(-1, sample_length * freq, 1))
            del batch_x, batch_y_1, batch_y_2, noise_E1, signal_E1_1, signal_E1_2, noise_E2, signal_E2_1, signal_E2_2, noise_E3, signal_E3_1, signal_E3_2
            gc.collect()
        count = count + 3
        if count > batch_size:
            count = 1
def max_and_min_peak(sample_length,snr_low_list,snr_high_list, peak_range,freq, snr_change_time, datapath):
    my_denoising_iter = detach_sample_cut_iter(sample_length, snr_low_list, snr_high_list, peak_range, freq,
                                               snr_change_time, datapath)
    max_signala=np.inf
    min_signala=np.inf
    max_signalb=np.inf
    min_signalb=np.inf
    n=0
    for noise_E1, signal_E1_1, signal_E1_2, noise_E2, signal_E2_1, signal_E2_2, noise_E3, signal_E3_1, signal_E3_2 in my_denoising_iter:
        # print(noise_E1.shape)
        # print(strain.shape)
        min_signala=min(min_signala,np.max(signal_E1_1),np.max(signal_E2_1),np.max(signal_E3_1))
        # min_signala=min(min_signala,np.min(signal_E1_1),np.min(signal_E2_1),np.min(signal_E3_1))
        min_signalb = min(min_signalb, np.max(signal_E1_2), np.max(signal_E2_2), np.max(signal_E3_2))
        # min_signalb = min(min_signalb, np.min(signal_E1_2), np.min(signal_E2_2), np.min(signal_E3_2))
        n+=1
        print(n)
    # return max_signala,min_signala,max_signalb,min_signalb
    return min_signala,min_signalb
# val_datapath = 'H://10-80_10-80'
# sample_length = 4
# snr_low_list = [30, 26, 22, 17, 12, 12, 17, 22, 26]
# snr_high_list = [26, 22, 17, 12, 8, 8, 12, 17, 22]
#     # snr_list=[50,45,40,35]
# peak_range = (0.5, 0.95)
# freq = 4096
# snr_change_time = 2000
# # batch_size =16
# min_signala,min_signalb = max_and_min_peak(sample_length=sample_length, snr_low_list=snr_low_list,
#                                  snr_high_list=snr_high_list, peak_range=peak_range, freq=freq,
#                                  snr_change_time=snr_change_time, datapath=val_datapath)
# # print('signala最大幅度',max_signala)
# print('signala最小幅度',min_signala)
# # print('signalb最大幅度',max_signalb)
# print('signalb最小幅度',min_signalb)
# n=0
# # for x,y1,y2 in generator:
# #     print(x.shape)
# #     print(y1.shape)
# #     print(y2.shape)
# #     n+=1
# #     print(n)
# for x,y in generator:
#     print(x.shape)
#     print(y.shape)
#     n+=1
#     print(n)