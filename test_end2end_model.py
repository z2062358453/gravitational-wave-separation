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
        situation1 = 0
        situation2 = 0
        situation3 = 0
        situation4 = 0
        overa1=0
        overa2=0
        overa3=0
        overa4 = 0
        overa5 = 0
        overa6 = 0
        overa7 = 0
        overa8 = 0
        overa9 = 0
        overb1 = 0
        overb2 = 0
        overb3 = 0
        overb4 = 0
        overb5 = 0
        overb6 = 0
        overb7 = 0
        overb8 = 0
        overb9 = 0
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
        test_data1 = get_test_iter_for_peak_example(SNRA=10,
                                                   SNRB=10,
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
            for egs, y1,y2 in test_data3:
                start_time=time.time()
                y1=torch.from_numpy(y1)
                y2=torch.from_numpy(y2)
                if torch.argmax(y1)<=torch.argmax(y2):
                   temp = y1.clone()
                   y1.copy_(y2)
                   y2.copy_(temp)
                egs=torch.from_numpy(egs)
                egs=egs.to(self.device)
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
                end_time=time.time()
                total_time=end_time-start_time
                # print('推理时间:',total_time)
                # mix_filename=mix_folder+f'_count{m}.png'
                # plt.figure(figsize=(8, 6))
                # plt.plot(egs.cpu().numpy().reshape(8192))
                # plt.title('Mix Signal')
                # if m % 100 == 0:
                #     plt.savefig(mix_filename)
                # plt.close()
                # index=0
                # for s in spks:
                #     s = s[:egs.shape[1]]
                #     s = s - torch.mean(s)
                #     s = s/torch.max(torch.abs(s))
                #     #norm
                #     #s = s*norm/torch.max(torch.abs(s))
                #     s = s.unsqueeze(0)
                #     index += 1
                #     # os.makedirs(file_path+'/singnal'+str(index), exist_ok=True)
                #     # Saparate_filename = file_path + '/singnal' + str(index) + '/' + f'_spk{index}_count{m}.png'
                #     if  index==1:
                #          Saparate_filename=file_path+'/singnal1/'+f'_spk{index}_count{m}.png'
                #     else:
                #          Saparate_filename = file_path + '/singnal2/' + f'_spk{index}_count{m}.png'
                #     print("overlap is :")
                #     print(overlap(s.cpu().numpy(),y1.cpu().numpy()if index==1 else -(y2.cpu().numpy())))
                #     # 绘制预测结果并保存
                #     plt.figure(figsize=(8, 6))
                #     plt.plot(s.cpu().numpy().reshape(freq*sample_length))
                #     plt.plot(y1.cpu().numpy().reshape(freq*sample_length) if index==1 else -(y2.cpu().numpy()).reshape(freq*sample_length))
                #     # plt.plot(y1.reshape(8192) if index == 1 else y2.reshape(8192))
                #     plt.title('Saparate_Signal')
                #     # plt.plot(s2.cpu().numpy().reshape(8192))
                #     # plt.plot(y2.reshape(8192))
                #     # plt.legend()
                #     # plt.title('Saparate_Signal')
                #     if m % 100 == 0:
                #         plt.savefig(Saparate_filename)
                #     plt.show()
                #     plt.close()
                # m = m + 1
                for i in range(0, len(spks), 2):
                    s1 = spks[i]
                    s2 = spks[i + 1]
                    s1 = s1[:egs.shape[1]]
                    s1 = s1 - torch.mean(s1)
                    s1 = s1 / torch.max(torch.abs(s1))
                    s2 = s2[:egs.shape[1]]
                    s2 = s2 - torch.mean(s2)
                    s2 = s2 / torch.max(torch.abs(s2))
                    s1 = s1.unsqueeze(0)
                    s2 = s2.unsqueeze(0)
                    # if torch.argmax(s1) >= torch.argmax(s2):
                    #     temp = s1.clone()
                    #     s1.copy_(s2)
                    #     s2.copy_(temp)
                    overlap1=overlap(s1.cpu().numpy(), y1.cpu().numpy())
                    overlap2=overlap(s2.cpu().numpy(), -y2.cpu().numpy())
                    if overlap1>=0.9 and overlap2>=0.9:
                        situation1+=1
                    elif overlap1>=0.9 and overlap2<0.9:
                        situation2+=1
                        if  0.8<overlap2<0.9:
                            overa1+=1
                        elif 0.7<overlap2<0.8:
                            overa2 += 1
                        elif 0.6 < overlap2 < 0.7:
                            overa3 += 1
                        elif 0.5 < overlap2 < 0.6:
                            overa4 += 1
                        elif 0.4 < overlap2 < 0.5:
                            overa5 += 1
                        elif 0.3 < overlap2 < 0.4:
                            overa6 += 1
                        elif 0.2 < overlap2 < 0.3:
                            overa7 += 1
                        elif 0.1 < overlap2 < 0.2:
                            overa8 += 1
                        else:
                            overa9+=1
                    elif overlap1<0.9 and overlap2>=0.9:
                        situation3+=1
                        if 0.8 < overlap1 < 0.9:
                            overb1 += 1
                        elif 0.7 < overlap1 < 0.8:
                            overb2 += 1
                        elif 0.6 < overlap1 < 0.7:
                            overb3 += 1
                        elif 0.5 < overlap1 < 0.6:
                            overb4 += 1
                        elif 0.4 < overlap1 < 0.5:
                            overb5 += 1
                        elif 0.3 < overlap1 < 0.4:
                            overb6 += 1
                        elif 0.2 < overlap1 < 0.3:
                            overb7 += 1
                        elif 0.1 < overlap1 < 0.2:
                            overb8 += 1
                        else:
                            overb9 += 1
                    else:
                        situation4+=1
                    if overlap1 < 0.9 and overlap2 < 0.9:
                        temp = y1.clone()
                        y1.copy_(y2)
                        y2.copy_(temp)
                        # print("overlap for signal 1:")
                        # print(overlap(s1.cpu().numpy(), y1.cpu().numpy()))
                        # print("overlap for signal 2:")
                        # print(overlap(s2.cpu().numpy(), -y2.cpu().numpy()))
                        overlap3 = overlap(s1.cpu().numpy(), y1.cpu().numpy())
                        overlap4 = overlap(s2.cpu().numpy(), -y2.cpu().numpy())
                        if overlap3 >= 0.9 and overlap4 >= 0.9:
                            situation1 += 1
                            situation4-=1
                        elif overlap3 >= 0.9 and overlap4 < 0.9:
                            # print("交换后overlap2小于0.9")
                            # print(overlap4)
                            situation3 += 1
                            situation4 -= 1
                            if 0.8 < overlap4 < 0.9:
                                overa1 += 1
                            elif 0.7 < overlap4 < 0.8:
                                overa2 += 1
                            elif 0.6 < overlap4 < 0.7:
                                overa3 += 1
                            elif 0.5 < overlap4 < 0.6:
                                overa4 += 1
                            elif 0.4 < overlap4 < 0.5:
                                overa5 += 1
                            elif 0.3 < overlap4 < 0.4:
                                overa6 += 1
                            elif 0.2 < overlap4 < 0.3:
                                overa7 += 1
                            elif 0.1 < overlap4 < 0.2:
                                overa8 += 1
                            else:
                                overa9 += 1
                        elif overlap3 < 0.9 and overlap4 >= 0.9:
                            # print("交换后overlap1小于0.9")
                            # print(overlap3)
                            situation2 += 1
                            situation4 -= 1
                            if 0.8 < overlap3 < 0.9:
                                overb1 += 1
                            elif 0.7 < overlap3 < 0.8:
                                overb2 += 1
                            elif 0.6 < overlap3 < 0.7:
                                overb3 += 1
                            elif 0.5 < overlap3 < 0.6:
                                overb4 += 1
                            elif 0.4 < overlap3 < 0.5:
                                overb5 += 1
                            elif 0.3 < overlap3 < 0.4:
                                overb6 += 1
                            elif 0.2 < overlap3 < 0.3:
                                overb7 += 1
                            elif 0.1 < overlap3 < 0.2:
                                overb8 += 1
                            else:
                                overb9 += 1
                        # if overlap3 < 0.9 and overlap4 < 0.9:
                        #     if  overlap3>=0.65 and overlap4>=0.65:
                        #             print('交换后overlap1和2全大于0.65')
                        #             print(overlap3)
                        #             print(overlap4)
                        #     elif overlap3>=0.65 and overlap4<0.65:
                        #             print('交换后overlap1大于0.65')
                        #             print(overlap3)
                        #             print(overlap4)
                        #     elif overlap3<0.65 and overlap4>=0.65:
                        #             print('交换后overlap2大于0.65')
                        #             print(overlap3)
                        #             print(overlap4)
                            # if overlap3<0.65 and overlap4<0.65:
                            #     temp = y1.clone()
                            #     y1.copy_(y2)
                            #     y2.copy_(temp)
                                # overlap5 = overlap(s1.cpu().numpy(), y1.cpu().numpy())
                                # overlap6 = overlap(s2.cpu().numpy(), -y2.cpu().numpy())
                                # print('再次交换后的')
                                # print(overlap5)
                                # print(overlap6)
                                # plt.figure(figsize=(8, 6))
                                # plt.plot(s1.cpu().numpy().reshape(freq * sample_length))
                                # plt.plot(y1.cpu().numpy().reshape(freq * sample_length))
                                # plt.title('Separate Signal 1')
                                # plt.show()
                                # plt.close()
                                # # 绘制并保存信号2
                                # plt.figure(figsize=(8, 6))
                                # plt.plot(s2.cpu().numpy().reshape(freq * sample_length))
                                # plt.plot(-y2.cpu().numpy().reshape(freq * sample_length))
                                # plt.title('Separate Signal 2')
                                # plt.show()
                                # plt.close()
        print('交换前overlap2在0.8-0.9：',overa1)
        print('交换前overlap2在0.7-0.8：', overa2)
        print('交换前overlap2在0.6-0.7：', overa3)
        print('交换前overlap2在0.5-0.6：', overa4)
        print('交换前overlap2在0.4-0.5：', overa5)
        print('交换前overlap2在0.3-0.4：', overa6)
        print('交换前overlap2在0.2-0.3：', overa7)
        print('交换前overlap2在0.1-0.2：', overa8)
        print('交换前overlap2在0-0.1：', overa9)

        print('交换前overlap1在0.8-0.9：', overb1)
        print('交换前overlap1在0.7-0.8：', overb2)
        print('交换前overlap1在0.6-0.7：', overb3)
        print('交换前overlap1在0.5-0.6：', overb4)
        print('交换前overlap1在0.4-0.5：', overb5)
        print('交换前overlap1在0.3-0.4：', overb6)
        print('交换前overlap1在0.2-0.3：', overb7)
        print('交换前overlap1在0.1-0.2：', overb8)
        print('交换前overlap1在0-0.1：', overb9)

        print('A和B都分离：',situation1)
        print('只有A分离：',situation2)
        print('只有B分离：',situation3)
        print('A和B都没分离：',situation4)
                    # print("overlap for signal 1:")
                    # print(overlap(s1.cpu().numpy(), y1.cpu().numpy()))
                    # print("overlap for signal 2:")
                    # print(overlap(s2.cpu().numpy(), -y2.cpu().numpy()))
                #     Saparate_filename1 = f"{file_path}/singnal1/_spk1_count{m}.png"
                #     Saparate_filename2 = f"{file_path}/singnal2/_spk2_count{m}.png"  # 绘制并保存信号1
                #     plt.figure(figsize=(8, 6))
                #     plt.plot(s1.cpu().numpy().reshape(freq * sample_length))
                #     plt.plot(y1.cpu().numpy().reshape(freq * sample_length))
                #     plt.title('Separate Signal 1')
                #     if m % 100 == 0:
                #         plt.savefig(Saparate_filename1)
                #     plt.show()
                #     plt.close()
                #     # 绘制并保存信号2
                #     plt.figure(figsize=(8, 6))
                #     plt.plot(s2.cpu().numpy().reshape(freq * sample_length))
                #     plt.plot(-y2.cpu().numpy().reshape(freq * sample_length))
                #     plt.title('Separate Signal 2')
                #     if m % 100 == 0:
                #         plt.savefig(Saparate_filename2)
                #     plt.show()
                #     plt.close()
                # m += 1
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