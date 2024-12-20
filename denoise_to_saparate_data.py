import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gc
import os
import denoise_pytorch_trainer
from config.option1 import parse
import argparse
from load_data_saparate import get_train_batch_for_denoise_task
from load_data_saparate import get_val_batch_for_denoise_task
from load_data_zwg import get_train_batch_iter
from load_data_saparate_fudu import get_train_batch_iter_saparate
from matplotlib import pyplot as plt
import matplotlib
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
        print('noiseE1, noiseE2, noiseE3 are 128 s length noise of E1, E2 and E3')
        print('signal_E1_list, signal_E2_list and signal_E3_list are signals, and each have 1 samples')
        print('mass1_list, mass2_list are masses, and each have 1 masses')
        print('right_ascension_list and declination_list are the directions of the source of the signal')

def get_saparate_data_train():
    model = denoise_pytorch_trainer.MyModel()
    model_path = './checkpoint_mse/MyModel/best_fudu.pt'
    dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(dict["model_state_dict"])
    model = model.cuda()
    model.eval()

    # 设置批处理大小
    saparate_batch_size = 4

    train_datapath = 'H://10-80_10-80'
    sample_length = 4
    snr_low_list = [30, 26, 22, 17, 12, 12, 17, 22, 26]
    snr_high_list = [26, 22, 17, 12, 8, 8, 12, 17, 22]
    peak_range = (0.5, 0.95)
    freq = 4096
    snr_change_time = 2000

    # 使用 Torch 的 DataLoader 管理数据加载和批处理
    data_loader = get_train_batch_iter_saparate(batch_size=saparate_batch_size, sample_length=sample_length,
                                                 snr_low_list=snr_low_list, snr_high_list=snr_high_list,
                                                 peak_range=peak_range, freq=freq, snr_change_time=snr_change_time,
                                                 datapath=train_datapath)

    for x, y1, y2 in data_loader:
        x = torch.from_numpy(x)
        x = x.cuda()

        # 在 GPU 上进行推理
        with torch.no_grad():
            y = model(x)

        # 将推理结果转移到 CPU 上
        y = y.cpu().numpy()

        yield y.reshape(-1, sample_length * freq, 1), y1, y2
# def get_saparate_data_val():
#     # opt = parse('./config/MyModel/train1.yml')
#     model = denoise_pytorch_trainer.MyModel()
#     model_path = 'E://Dual-Path-RNN-Pytorch-master//checkpoint_mse//MyModel//best.pt'
#     dict=torch.load(model_path, map_location='cuda:0')
#     model.load_state_dict(dict["model_state_dict"])
#     model.eval()
#     batch_x=[]
#     batch_y1=[]
#     batch_y2=[]
#     test_path='H://val'
#     sample_length = 4
#     snr_low_list = [30, 26, 22, 17, 12, 12, 17, 22, 26]
#     snr_high_list = [26, 22, 17, 12, 8, 8, 12, 17, 22]
#     peak_range = (0.5, 0.95)
#     freq = 2048
#     m=1
#     snr_change_time = 2000
#     batch_size = 1
#     saparate_batch_size=16
#     data = get_val_batch_for_denoise_task(batch_size=batch_size,sample_length=sample_length,snr_low_list=snr_low_list,snr_high_list=snr_high_list, peak_range=peak_range,freq=freq, snr_change_time=snr_change_time, datapath=test_path)
#     for x, y1, y2 in data:
#         # x为带噪声的混合信号,y为去噪后的结果,y1,y2分别为两个标签
#         x = torch.from_numpy(x)
#         y = model(x)
#         y = y.detach().numpy()
#         if m == 1:
#             batch_x = y
#             batch_y1 = y1
#             batch_y2 = y2
#         else:
#             batch_x = np.concatenate(
#                 (batch_x, (y)))
#             batch_y1 = np.concatenate(
#                 (batch_y1, (y1)))
#             batch_y2 = np.concatenate(
#                 (batch_y2, (y2)))
#         if m == saparate_batch_size:
#             yield (batch_x.reshape(-1, sample_length * freq, 1), batch_y1.reshape(-1, sample_length * freq, 1),
#                    batch_y2.reshape(-1, sample_length * freq, 1))
#             del batch_x, batch_y1, batch_y2
#             gc.collect()
#         m += 1
#         # You can now use `y` for further processing
#         if m > saparate_batch_size:
#             m = 1
def get_saparate_data_test():
    # 加载模型并移动到 GPU 上
    model = denoise_pytorch_trainer.MyModel()
    model_path = './checkpoint_mse/MyModel/best_fudu.pt'
    dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(dict["model_state_dict"])
    model = model.cuda()
    model.eval()

    # 设置批处理大小
    saparate_batch_size = 1

    test_path = 'H://test'
    sample_length = 4
    snr_low_list = [30, 26, 22, 17, 12, 12, 17, 22, 26]
    snr_high_list = [26, 22, 17, 12, 8, 8, 12, 17, 22]
    peak_range = (0.5, 0.95)
    freq = 4096
    snr_change_time = 2000

    # 使用 Torch 的 DataLoader 管理数据加载和批处理
    data_loader = get_train_batch_iter_saparate(batch_size=saparate_batch_size, sample_length=sample_length,
                                                 snr_low_list=snr_low_list, snr_high_list=snr_high_list,
                                                 peak_range=peak_range, freq=freq, snr_change_time=snr_change_time,
                                                 datapath=test_path)

    for x, y1, y2 in data_loader:
        x = torch.from_numpy(x)
        x = x.cuda()

        # 在 GPU 上进行推理
        with torch.no_grad():
            y = model(x)

        # 将推理结果转移到 CPU 上
        y = y.cpu().numpy()

        yield y.reshape(-1, sample_length * freq, 1), y1, y2
def get_saparate_data_val():
        # 加载模型并移动到 GPU 上
        model = denoise_pytorch_trainer.MyModel()
        model_path = './checkpoint_mse/MyModel/best_fudu.pt'
        dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(dict["model_state_dict"])
        model = model.cuda()
        model.eval()

        # 设置批处理大小
        saparate_batch_size = 4

        test_path = 'H://val'
        sample_length = 4
        snr_low_list = [30, 26, 22, 17, 12, 12, 17, 22, 26]
        snr_high_list = [26, 22, 17, 12, 8, 8, 12, 17, 22]
        peak_range = (0.5, 0.95)
        freq = 4096
        snr_change_time = 2000

        # 使用 Torch 的 DataLoader 管理数据加载和批处理
        data_loader = get_train_batch_iter_saparate(batch_size=saparate_batch_size, sample_length=sample_length,
                                                     snr_low_list=snr_low_list, snr_high_list=snr_high_list,
                                                     peak_range=peak_range, freq=freq, snr_change_time=snr_change_time,
                                                     datapath=test_path)

        for x, y1, y2 in data_loader:
            x = torch.from_numpy(x)
            x = x.cuda()

            # 在 GPU 上进行推理
            with torch.no_grad():
                y = model(x)

            # 将推理结果转移到 CPU 上
            y = y.cpu().numpy()

            yield y.reshape(-1, sample_length * freq, 1), y1, y2
def get_saparate_data_val1():
    opt = parse('./config/MyModel/train1.yml')
    model = denoise_pytorch_trainer.MyModel()
    model_path = 'E://Dual-Path-RNN-Pytorch-master//checkpoint_mse//MyModel//best.pt'
    dict=torch.load(model_path, map_location='cpu')
    model.load_state_dict(dict["model_state_dict"])
    batch_x=[]
    batch_y1=[]
    batch_y2=[]
    test_path='H://test'
    sample_length = 4
    snr_low_list = [30, 26, 22, 17, 12, 12, 17, 22, 26]
    snr_high_list = [26, 22, 17, 12, 8, 8, 12, 17, 22]
    # snr_list=[50,45,40,35]
    peak_range = (0.5, 0.95)
    freq = 4096
    m=1
    snr_change_time = 2000
    batch_size = 1
    data =get_train_batch_iter(batch_size=batch_size,sample_length=sample_length,snr_low_list=snr_low_list,snr_high_list=snr_high_list, peak_range=peak_range,freq=freq, snr_change_time=snr_change_time, datapath=test_path)
    # print(f'data的长度：{len(data)}')
    output_folder = './save_signals_denoise/'
    os.makedirs(output_folder, exist_ok=True)
    for x,y1 in data:
        x=torch.from_numpy(x)
        # x[0]:噪声+信号    X[1]：真实信号    y:去噪后的信号（预测信号）
        model.eval()
        y = model(x)
        y=y.detach().numpy()
        print(y.shape)


        # 绘制预测结果并保存
        plt.figure(figsize=(8, 6))
        # plt.plot(y.reshape(1024), 'y-.')
        plt.plot(y.reshape(sample_length*freq), 'y-.')
        plt.title('Predicted Signal')
        if m % 100 == 0:
            plt.savefig(os.path.join(output_folder + 'predicted_signal', "predicted_signal{}.png".format(m + 1)))
        plt.close()

        # 合并信号
        plt.figure()
        # 绘制原始信号
        # plt.plot(x[1].reshape(1024), label='Original Signal')
        plt.plot(y1.reshape(sample_length*freq), label='Original Signal')
        # 绘制预测信号
        # plt.plot(y.reshape(1024), 'y-.', label='Predicted Signal')
        plt.plot(y.reshape(sample_length*freq), 'y-.', label='Predicted Signal')
        # 添加图例
        plt.legend()
        # 设置图表标题
        plt.title('Comparison of Original and Predicted Signals')
        # 保存图表到指定文件夹中
        if m % 100 == 0:
            plt.savefig(os.path.join(output_folder + 'combine_signal', "combine_signal{}.png".format(m + 1)))
        # 显示图表
        plt.show()
        plt.close()

        m = m + 1
#
# generator = get_saparate_data_val()
# n=0
# for x,y1,y2 in generator:
#     print(x.shape)
#     print(y1.shape)
#     print(y2.shape)
#     n+=1
#     print(n)