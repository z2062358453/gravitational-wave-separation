import os
import torch
import argparse
import denoise_pytorch_trainer
import end_to_end_denoise_saparate_trainer
from model.model import Conv_TasNet
import argparse
import torch
from config.option import parse
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
import matplotlib
from load_data_for_saparate_test import  get_test_iter_for_snr_peak_time_example
from load_data_for_saparate_test import  get_test_iter_for_snr_example
from load_data_for_saparate_test import  get_test_iter_for_peak
from load_data_for_saparate_test import  get_test_iter_for_snr
import time
import os
import matplotlib.pyplot as plt
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
yaml_path = './config/Conv_Tasnet/train.yml'
opt = parse(yaml_path)
Conv_Tasnet = Conv_TasNet(**opt['Conv_Tasnet'])
model_path = './checkpoint/Conv_Tasnet/best.pt'
dict2 = torch.load(model_path, map_location='cpu')
Conv_Tasnet.load_state_dict(dict2["model_state_dict"])
Conv_Tasnet = Conv_Tasnet.cuda()

yaml_path1 = './config/Dual_RNN/train_rnn.yml'
opt = parse(yaml_path1)
Dual_Path_RNN= Dual_RNN_model(**opt['Dual_Path_RNN'])
model_path = './checkpoint/Dual_Path_RNN/best_rnn.pt'
dict3 = torch.load(model_path, map_location='cuda:0')
Dual_Path_RNN.load_state_dict(dict3["model_state_dict"])
Dual_Path_RNN = Dual_Path_RNN.cuda()
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

    def inference(self,file_path):
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
        test_data = get_test_iter_for_snr_peak_time_example(SNRA=10,
                                                            SNRB=10,
                                                            sample_length=sample_length,
                                                            freq=freq,
                                                            datapath=test_datapath)
        fig,axes=plt.subplots(8,2,figsize=(24,34))
        # test_data = get_saparate_data_test()
        xuhao=['a','b','c','d','e','f','g','h','i','j','k']
        for i,(peak_time_x,snr_x) in enumerate(test_data):
            # for egs, y1, y2 in test_data1:
            #     start_time=time.time()
            #     # y0 =torch.from_numpy(y0)
            #     # noise = torch.from_numpy(noise)
            #     # signal1 = torch.from_numpy(signal1)
            #     # signal2 = torch.from_numpy(signal2)
            #     y1=torch.from_numpy(y1)
            #     y2=torch.from_numpy(y2)
            #     if torch.argmax(y1)>=torch.argmax(y2):
            #        temp = y1.clone()
            #        y1.copy_(y2)
            #        y2.copy_(temp)

                # peak_time_y=torch.from_numpy(peak_time_y)
                # peak_time_y=peak_time_y.to(self.device)
                # norm = torch.norm(peak_time_y,float('inf'))
                # if len(self.gpuid) != 0:
                #     if peak_time_y.dim() == 1:
                #         peak_time_y = torch.unsqueeze(peak_time_y, 0)
                #     peak_time_ests=self.net(peak_time_y)
                #     spks=[torch.squeeze(s.detach().cpu()) for s in  peak_time_ests]
                # else:
                #     if peak_time_y.dim() == 1:
                #         peak_time_y = torch.unsqueeze(peak_time_x, 0)
                #     peak_time_ests=self.net(peak_time_y)
                #     spks=[torch.squeeze(s.detach()) for s in peak_time_ests]
                # snr_y = torch.from_numpy(snr_y)
                # snr_y = snr_y.to(self.device)
                # norm = torch.norm(snr_y, float('inf'))
                # if len(self.gpuid) != 0:
                #     if snr_y.dim() == 1:
                #         snr_y = torch.unsqueeze(snr_y, 0)
                #     peak_time_ests1 = self.net(snr_y)
                #     spks1 = [torch.squeeze(s1.detach().cpu()) for s1 in peak_time_ests1]
                # else:
                #     if snr_y.dim() == 1:
                #         snr_y = torch.unsqueeze(snr_y, 0)
                #     peak_time_ests1 = self.net(snr_y)
                #     spks1 = [torch.squeeze(s1.detach()) for s1 in peak_time_ests1]
                # # end_time=time.time()
                # # total_time=end_time-start_time
                # # print('推理时间:',total_time)
                #
                # # for i in range(0, len(spks), 2):
                # s1 = spks[0]
                # s2 = spks[1]
                # s1 = s1[:peak_time_y.shape[1]]
                # s1 = s1 - torch.mean(s1)
                # s1 = s1 / torch.max(torch.abs(s1))
                # s2 = s2[:peak_time_y.shape[1]]
                # s2 = s2 - torch.mean(s2)
                # s2 = s2 / torch.max(torch.abs(s2))
                # s1 = s1.unsqueeze(0)
                # s2 = s2.unsqueeze(0)
                # s3 = spks1[0]
                # s4 = spks1[1]
                # s3 = s3[:snr_y.shape[1]]
                # s3 = s3 - torch.mean(s3)
                # s3 = s3 / torch.max(torch.abs(s3))
                # s4 = s4[:snr_y.shape[1]]
                # s4 = s4 - torch.mean(s4)
                # s4 = s4 / torch.max(torch.abs(s4))
                # s3 = s3.unsqueeze(0)
                # s4 = s4.unsqueeze(0)
                ax1=axes[i,0]
                ax2=axes[i,1]
                color2='green'
                color1='red'
                color3 = 'blue'
                color4 = 'orange'
                # peak_time_x=peak_time_x[4096*2.5:-1]
                # snr_x=snr_x[4096*2.5:-1]
                t=np.linspace(0,sample_length,sample_length*freq)
                ax1.plot(t,peak_time_x.reshape(freq * 4),color=color3)
                # ax1.plot(t,peak_SignalB.reshape(freq * sample_length),color=color3,label='Signal B')
                # ax1.plot(t,(s1+s2).reshape(freq * sample_length), color=color4,linestyle=':', label='Signal A')
                # ax1.plot(t,noise.reshape(freq * sample_length),color='gray',alpha=0.7,label='Noise')
                # ax1.set_title(f'({xuhao[i]}) SNR A:10,SNR B:{round(6 + 2 * i, 1)}',fontsize=11)
                ax1.set_title(f'(a{i+1}) Peak time difference:{round(0 - 0.1 * i, 1)}s',fontsize=21,fontname='Times New Roman')
                ax1.set_ylabel("Strain",fontsize=21,fontname='Times New Roman')
                # ax1.set_xlabel("t (s)",loc='right')
                # ax1.set_xticks(np.arange(0,4,4096))
                # ax1.set_xlim(0, 4)
                # ax1.set_ylim(-3.3e-23,3.3e-23)
                ax1.set_ylim(-3.12e-23,3.12e-23)
                # ax1.set_ylim(0,freq//2)
                ax1.set_xlim(2.5,4)
                # ax1.set_yticks([-1.8e-23,0,1.8e-23])
                axes[7,0].set_xlabel("Time (s)",labelpad=5,loc='center',fontsize=21,fontname='Times New Roman')
                # ax2.pcolormesh(t2, f2,Zxx2**0.5)
                ax2.plot(t, snr_x.reshape(freq * 4), color=color3)
                # ax2.plot(t, snr_SignalA.reshape(freq * sample_length), color=color4, label='Signal A')
                # ax2.plot(t, snr_SignalB.reshape(freq * sample_length), color=color3, label='Signal B')
                # ax2.plot(t,(s3+s4).cpu().numpy().reshape(freq * sample_length),color=color2,linestyle=':',label='Separated Signal B')
                ax2.set_title(f'(b{i+1}) SNR A:10,SNR B:{round(6 + 2 * i, 1)}',fontsize=21,fontname='Times New Roman')
                ax2.set_xlim(2.5, 4)
                axes[7, 1].set_xlabel("Time (s)", labelpad=5,loc='center',fontsize=21,fontname='Times New Roman')
                # ax2.set_xlabel("t (s)",loc='right')
                ax2.set_ylim(-3.4e-23,3.4e-23)
                # ax2.set_ylim(0, freq // 2)
                ax1.tick_params(axis="both", which='major', labelsize=20)
                ax2.tick_params(axis="both", which='major', labelsize=20)
                ax1.yaxis.offsetText.set_fontsize(20)
                ax2.yaxis.offsetText.set_fontsize(20)
                # ax2.plot(t,y1.cpu().numpy().reshape(freq * sample_length),color=color3,label='Signal A')
                # ax2.set_title(f'Overlap B:{overlap1}',fontsize=11)
                # axes[0,1].legend(framealpha=0.7,prop={"size":11},loc='upper right',fontsize=12,bbox_to_anchor=(1,1.1))
            # handles, labels = [], []
            # for ax in axes.flat:
            #         for handle, label in zip(*ax.get_legend_handles_labels()):
            #             if label not in labels:
            #                 handles.append(handle)
            #                 labels.append(label)
                # fig.legend(handles, labels, loc='upper center', framealpha=0.7, prop={"size": 10}, bbox_to_anchor=(0.928, 1),ncol=3)
        plt.tight_layout()
        plt.subplots_adjust()
        plt.savefig('./hunxiao1.pdf')
        # plt.show()
        plt.close()

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