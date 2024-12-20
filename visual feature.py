import os
import torch
import argparse
import denoise_pytorch_trainer
from model.model import Conv_TasNet
import argparse
import torch
# from config.option import parse
from torch.nn.parallel import data_parallel
from model.model import Conv_TasNet
from model.model_rnn import Dual_RNN_model
from logger.set_logger import setup_logger
import logging
import numpy as np
from config.option import parse
import matplotlib.pyplot as plt
from load_data_saparate import get_val_batch_for_end2end
from denoise_to_saparate_data import get_saparate_data_test
import end_to_end_denoise_saparate_trainer
import matplotlib
from load_data_for_saparate_test import  get_test_iter_for_peak_example
from load_data_for_saparate_test import  get_test_iter_for_snr_example
from load_data_for_saparate_test import  get_test_iter_for_peak
from load_data_for_saparate_test import  get_test_iter_for_snr
import time
from scipy.cluster.hierarchy import dendrogram,linkage
from matplotlib.colors import LogNorm,PowerNorm
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LogFormatter
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
# 加载第一个模型
model_denoise = denoise_pytorch_trainer.MyModel()
model_path = './checkpoint_mse/MyModel/best.pt'
dict1 = torch.load(model_path, map_location='cpu')
model_denoise.load_state_dict(dict1["model_state_dict"])
model_denoise = model_denoise.cuda()
def overlap(signala,signalb):
    signala=signala.reshape(4*4096)
    signalb=signalb.reshape(4*4096)
    # print(signala.shape)
    # print(signalb.shape)
    return np.sum(signala*signalb,axis=0)/np.sqrt(np.sum(signala*signala,axis=0))/np.sqrt(np.sum(signalb*signalb,axis=0))
# 加载第二个模型
# yaml_path = './config/Conv_Tasnet/train.yml'
# opt = parse(yaml_path)
# Conv_Tasnet = Conv_TasNet(**opt['Conv_Tasnet'])
# model_path = './checkpoint/Conv_Tasnet/best.pt'
# dict2 = torch.load(model_path, map_location='cpu')
# Conv_Tasnet.load_state_dict(dict2["model_state_dict"])
# Conv_Tasnet = Conv_Tasnet.cuda()

yaml_path1 = './config/Dual_RNN/train_rnn.yml'
opt = parse(yaml_path1)
Dual_Path_RNN= Dual_RNN_model(**opt['Dual_Path_RNN'])
model_path = './checkpoint/Dual_Path_RNN/best_rnn.pt'
dict3 = torch.load(model_path, map_location='cuda:0')
Dual_Path_RNN.load_state_dict(dict3["model_state_dict"])
Dual_Path_RNN = Dual_Path_RNN.cuda()
def get_feature_maps(model, layers, x):
    feature_maps = []

    def hook_fn(module, input, output):
        feature_maps.append(output)

    hooks = []
    for layer in layers:
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)
    with torch.no_grad():
        model(x)
    for hook in hooks:
        hook.remove()
    return feature_maps
def plot_basis_functions(basis_functions, title):
# 计算欧几里得距离并使用 UPGMA 方法排序
    Z = linkage(basis_functions, method='average', metric='euclidean')
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=10)
    ax.set_title(title)
    plt.show()
# 可视化特征图
def visualize_feature_maps(feature_maps, title):
    # num_subplots = min(8, len(feature_maps))
    num_subplots = 4
    fig, axes = plt.subplots(num_subplots, 1, figsize=(7, 7))
    if num_subplots == 1:
        axes = [axes]
    xuhao=['A','B']
    for i in range(num_subplots):
        feature_map = feature_maps[i].cpu().numpy()
        feature_map=feature_map.reshape(256,16383)
        feature_map_sum=np.sum(feature_map,axis=0)
        feature_map=(feature_map-feature_map.min())/(feature_map.max()-feature_map.min())
        im=axes[i].imshow(feature_map, aspect='auto', cmap='gray_r',norm=PowerNorm(gamma=0.3))
        t = np.linspace(0, 16382, 5)
        t_label=range(5)
        axes[i].set_xticks(t)
        axes[i].set_xticklabels(t_label)
        axes[i].set_ylim(0,256)
        axes[i].set_title(f'Mask{xuhao[i]}')
        plt.colorbar(im,ax=axes[i],orientation='vertical')
    plt.tight_layout()
    plt.savefig('./masks.png')
    plt.show()
def visualize_feature_maps1(feature_maps, s1,s2):
    # num_subplots = min(8, len(feature_maps))
    font = FontProperties(family='Times New Roman')
    num_subplots = 2
    fig, axes = plt.subplots(num_subplots, 2, figsize=(10, 4))
    if num_subplots == 1:
        axes = [axes]
    xuhao=['A','B']
    cmap=mcolors.LinearSegmentedColormap.from_list("custom_darkblue",["white","darkblue"])
    feature_map1 = feature_maps[1].cpu().numpy()
    feature_map1 = feature_map1.reshape(256, 16383)
    im1 = axes[0,0].imshow(feature_map1, aspect='auto', cmap=cmap, norm=PowerNorm(gamma=0.3))
    # feature_map1_sum = np.sum(feature_map1,axis=0)
    # axes[1].plot(feature_map1_sum)

    axes[1,0].plot(np.linspace(0,4,16384),s2.cpu().numpy().reshape(16384), color='black')

    feature_map2 = feature_maps[0].cpu().numpy()
    feature_map2 = feature_map2.reshape(256, 16383)
    im3 = axes[0,1].imshow(feature_map2, aspect='auto', cmap=cmap, norm=PowerNorm(gamma=0.3))

    # feature_map2_sum = np.sum(feature_map2, axis=0)
    # axes[4].plot(feature_map2_sum)

    axes[1,1].plot(np.linspace(0,4,16384),s1.cpu().numpy().reshape(16384), color='black')
    t = np.linspace(0, 16382, 5)
    t_label = range(5)
    axes[0,0].set_xticks(t)
    axes[0,0].set_xticklabels(t_label)
    axes[0,0].set_ylim(0, 256)
    axes[0,0].set_title('(a1) Mask A',fontsize=13,fontname='Times New Roman')

    # axes[1].set_xticks(t)
    # axes[1].set_xticklabels(t_label)
    # axes[1].set_yscale('log')
    # y_ticks=np.linspace(np.min( feature_map1_sum),np.max(feature_map2_sum),num=5)
    # y_label=[f'{np.log10(y_tick):.1f}' for y_tick in y_ticks]
    # axes[1].set_yticks(y_ticks)
    # axes[1].set_yticklabels(y_label)
    # axes[1].set_ylabel('Log Scale')
    # axes[1].set_title(f'Mask{xuhao[0]}_SUM')

    # axes[1].set_xticks(t)
    # axes[1].set_xticklabels(t_label)
    t1 = np.linspace(0, 4, 5)
    axes[1,0].set_xlim(0, 4)
    axes[1,0].set_xticks(t1)
    axes[1,0].set_xticklabels([int(tick) for tick in t1])
    axes[1,0].set_title('(a2) Signal A',fontsize=13,fontname='Times New Roman')
    axes[1, 0].set_xlabel("Time (s)", loc='center', fontsize=13,fontname='Times New Roman')

    axes[0,1].set_xticks(t)
    axes[0,1].set_xticklabels(t_label)
    axes[0,1].set_ylim(0, 256)
    axes[0,1].set_title('(b1) Mask B',fontsize=13,fontname='Times New Roman')

    # axes[4].set_xticks(t)
    # axes[4].set_xticklabels(t_label)
    # y_ticks1 = np.linspace(np.min(feature_map2_sum), np.max(feature_map2_sum), num=5)
    # y_label1 = [f'{np.log10(y_tick1):.1f}' for y_tick1 in y_ticks1]
    # axes[4].set_yticks(y_ticks1)
    # axes[4].set_yticklabels(y_label1)
    # axes[4].set_ylabel('Log Scale')
    # axes[3].set_yscale('log')
    # axes[3].set_ylim(0, 1)
    # axes[4].set_title(f'Mask{xuhao[1]}_SUM')
    # axes[4].set_yticks(np.arange(0, np.max(feature_map2_sum), 500))

    # axes[3].set_xticks(t)
    # axes[3].set_xticklabels(t_label)
    axes[1,1].set_xlim(0,4)
    t1 = np.linspace(0, 4, 5)
    axes[1,1].set_xlim(0, 4)
    axes[1,1].set_xticks(t1)
    axes[1,1].set_xticklabels([int(tick) for tick in t1])
    axes[1,1].set_title('(b2) Signal B',fontsize=13,fontname='Times New Roman')
    axes[1,1].set_xlabel("Time (s)", loc='center', fontsize=13,fontname='Times New Roman')

    # cbar_ax=fig.add_axes([0.90,0.62,0.02,0.29])
    # plt.colorbar(im1, ax=cbar_ax, orientation='vertical')
    # cbar=fig.colorbar(im1, cax=cbar_ax,orientation='vertical')
    # cbar.set_ticks(np.linspace(np.min(feature_map1),np.max(feature_map1),5))
    plt.tight_layout(rect=[0,0,1,1])
    # cbar = fig.colorbar(im1, ax=axes[0], orientation='vertical')
    # cbar.set_ticks(np.linspace(np.min(feature_map1), np.max(feature_map1), 5))
    plt.savefig('./masks.tif')
    plt.show()
def visualize_feature_maps2(feature_maps, title):
    # num_subplots = min(8, len(feature_maps))
    num_subplots = 2
    fig, axes = plt.subplots(num_subplots, 1, figsize=(7, 10))
    if num_subplots == 1:
        axes = [axes]
    xuhao=['A','B']
    feature_map1 = feature_maps[0].cpu().numpy()
    feature_map1 = feature_map1.reshape(256, 16383)
    # feature_map_sum = np.sum(feature_map, axis=0)
    # feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
    im1 = axes[0].imshow(feature_map1, aspect='auto', cmap='gray_r', norm=PowerNorm(gamma=0.3))

    feature_map2 = feature_maps[1].cpu().numpy()
    feature_map2 = feature_map2.reshape(256, 16383)
    im2 = axes[1].imshow(feature_map2, aspect='auto', cmap='gray_r', norm=PowerNorm(gamma=0.3))

    t = np.linspace(0, 16382, 5)
    t_label = range(5)
    axes[0].set_xticks(t)
    axes[0].set_xticklabels(t_label)
    axes[0].set_ylim(0, 256)
    axes[0].set_title(f'Mask{xuhao[0]}')
    axes[1].set_xticks(t)
    axes[1].set_xticklabels(t_label)
    axes[1].set_ylim(0, 0.52)
    axes[1].set_title(f'Mask{xuhao[1]}')
    # axes[2].set_xticks(t)
    # axes[2].set_xticklabels(t_label)
    # axes[2].set_ylim(0, 256)
    # axes[2].set_title(f'Mask{xuhao[1]}')
    # axes[3].set_xticks(t)
    # axes[3].set_xticklabels(t_label)
    # axes[3].set_ylim(0, 0.52)
    # axes[3].set_title(f'Mask{xuhao[1]}_SUM')
    axes[1].set_xlabel("time (s)", labelpad=3, loc='center', fontsize=15)
    # plt.colorbar(im, ax=axes[i], orientation='vertical')
    plt.tight_layout()
    plt.savefig('./masks.png')
    plt.show()
# 可视化特征图的函数
# def visualize_feature_maps(feature_maps):
#     for i, fmap in enumerate(feature_maps):
#         fmap = fmap[0] # 只取一个样本的特征图
#         fmap = fmap.cpu().numpy()
#         num_subplots=min(8,fmap.shape[0])
#         fig, axes = plt.subplots(num_subplots,1,figsize=(20,2*num_subplots))
#         if num_subplots==1:
#             axes=[axes]
#         for j in range(num_subplots):
#             axes[j].plot(fmap[j])
# #         plt.savefig("./feature_map.png")
class Separation():
    def __init__(self, yaml_path, model, gpuid):
        super(Separation, self).__init__()
        opt = parse(yaml_path)
        # net = Conv_TasNet(**opt['Conv_Tasnet'])
        # net = end_to_end_denoise_saparate_trainer.CombinedModel(model_denoise,Conv_Tasnet)
        net = end_to_end_denoise_saparate_trainer.CombinedModel(model_denoise, Dual_Path_RNN)
        dicts = torch.load(model, map_location='cpu')
        net.load_state_dict(dicts["model_state_dict"])
        # setup_logger(opt['logger']['name'], opt['logger']['path'],
        #                     screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
        self.logger = logging.getLogger(opt['logger']['name'])
        self.logger.info('Load checkpoint from {}, epoch {: d}'.format(model, dicts["epoch"]))
        self.net=net.cuda()
        self.device=torch.device('cuda:{}'.format(
            gpuid[0]) if len(gpuid) > 0 else 'cpu')
        self.gpuid=tuple(gpuid)

    def inference(self, file_path):
        os.makedirs(file_path,exist_ok=True)
        # mix_folder=os.path.join(file_path,'mix')
        # os.makedirs(mix_folder,exist_ok=True)
        m=0
        test_datapath = 'H://test'
        sample_length = 4
        snr_low_list = [30, 26, 22, 17, 12, 12, 17, 22, 26]
        snr_high_list = [26, 22, 17, 12, 8, 8, 12, 17, 22]
        # snr_list=[50,45,40,35]
        peak_range = (0.5, 0.95)
        freq = 4096
        snr_change_time = 2000
        batch_size = 1
        temp=0
        test_data1 = get_test_iter_for_peak_example(SNRA=12,
                                                   SNRB=12,
                                                   sample_length=sample_length,
                                                   freq=freq,
                                                   datapath=test_datapath)
        test_data2 = get_test_iter_for_snr_example(
                                                   sample_length=sample_length,
                                                   freq=freq,
                                                   datapath=test_datapath)
        test_data3=get_test_iter_for_peak(SNRA=10,
                                          SNRB=10,
                                          sample_length=sample_length,
                                          freq=freq,
                                          datapath=test_datapath)
        test_data4 = get_test_iter_for_snr(peak_time=(0.5,0.95),
                                           sample_length=sample_length,
                                           freq=freq,
                                           datapath=test_datapath)
        # test_data = get_saparate_data_test()
        with torch.no_grad():
            for i,(noise,signal1,signal2,egs,y1,y2) in enumerate(test_data1):
                start_time=time.time()
                y1=torch.from_numpy(y1)
                y2=torch.from_numpy(y2)
                # if torch.argmax(y1)<=torch.argmax(y2):
                #    temp = y1.clone()
                #    y1.copy_(y2)
                #    y2.copy_(temp)
                egs=torch.from_numpy(egs)
                egs=egs.to(self.device)
                y1=y1.to(self.device)
                y2 = y2.to(self.device)
                norm = torch.norm(egs,float('inf'))
                if len(self.gpuid) != 0:
                    if egs.dim() == 1:
                        egs = torch.unsqueeze(egs, 0)
                    ests=self.net(egs)
                    spks=[torch.squeeze(s.detach().cpu()) for s in ests]
                else:
                    if egs.dim() == 1:
                        egs = torch.unsqueeze(egs, 0)
                    ests=self.net(egs)
                    spks=[torch.squeeze(s.detach()) for s in ests]
                s1 = spks[0]
                s2 = spks[1]
                s1 = s1[:egs.shape[1]]
                s1 = s1 - torch.mean(s1)
                s1 = s1 / torch.max(torch.abs(s1))
                s2 = s2[:egs.shape[1]]
                s2 = s2 - torch.mean(s2)
                s2 = s2 / torch.max(torch.abs(s2))
                s1 = s1.unsqueeze(0)
                s2 = s2.unsqueeze(0)
                layer = [self.net.Dual_Path_RNN.separation]
                # layer = [self.net.Dual_Path_RNN.encoder]
                # layer = [self.net.Dual_Path_RNN.encoder, self.net.Dual_Path_RNN.separation]
                feature_maps = get_feature_maps(self.net, layer, egs)
                # decoder_feature=[feature_maps[0]*feature_maps[1][i] for i in range(2)]
                for i, feature_map in enumerate(feature_maps):
                    visualize_feature_maps1(feature_map, y2,y1)

                # visualize_feature_maps2(decoder_feature, f'Layer {i+1}')
                end_time=time.time()
                total_time=end_time-start_time
                # print('推理时间:',total_time)

                # index=0
                # for i in range(0, len(spks), 2):
                #     s1 = spks[i]
                #     s2 = spks[i + 1]
                #     s1 = s1[:egs.shape[1]]
                #     s1 = s1 - torch.mean(s1)
                #     s1 = s1 / torch.max(torch.abs(s1))
                #     s2 = s2[:egs.shape[1]]
                #     s2 = s2 - torch.mean(s2)
                #     s2 = s2 / torch.max(torch.abs(s2))
                #     s1 = s1.unsqueeze(0)
                #     s2 = s2.unsqueeze(0)
                #     # if torch.argmax(s1) >= torch.argmax(s2):
                #     #     temp = s1.clone()
                #     #     s1.copy_(s2)
                #     #     s2.copy_(temp)

def main():
    parser=argparse.ArgumentParser()
    # parser.add_argument(
    #     '-yaml', type=str, default='./config/Conv_Tasnet/train.yml', help='Path to yaml file.')
    parser.add_argument(
        '-yaml', type=str, default='./config/End_to_End_Model/train.yml', help='Path to yaml file.')
    parser.add_argument(
        '-model', type=str, default='./checkpoint_end2end/end_to_end_model/best_end2end.pt', help="Path to model file.")
    # parser.add_argument(
    #     '-model', type=str, default='./checkpoint/Dual_Path_RNN/best_end2end.pt', help="Path to model file.")
    parser.add_argument(
        '-gpuid', type=str, default='0', help='Enter GPU id number')
    # parser.add_argument(
    #     '-save_path', type=str, default='./Saparate_task_signal_end2end_Conv', help='save result path')
    parser.add_argument(
        '-save_path', type=str, default='./Saparate_task_signal_end2end_RNN', help='save result path')
    args=parser.parse_args()
    gpuid=[int(i) for i in args.gpuid.split(',')]
    separation=Separation(args.yaml, args.model, gpuid)
    separation.inference(args.save_path)


if __name__ == "__main__":
    main()