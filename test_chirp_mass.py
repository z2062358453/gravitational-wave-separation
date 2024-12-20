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
import pickle
import time
matplotlib.use('TkAgg')
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

def load_data_for_chirp_mass_test():

     with open('./new_test_chirpmass_data.pkl', 'rb') as f:
        loaded_data = pickle.load(f)  # 加载时，你得到的是一个元组，包含你之前保存的所有对象
     with open('./noise_list.pkl', 'rb') as f:
        noise = pickle.load(f)  # 加载时，你得到的是一个元组，包含你之前保存的所有对象
     noise=noise[0]
     chirp_mass_list_loaded, signal_list_loaded, param_loaded = loaded_data
     print(len(chirp_mass_list_loaded))
     for i in range(71):
         for j in range(i+1,71):
                mix_signal=(signal_list_loaded[i]+signal_list_loaded[j]+noise)/np.max(signal_list_loaded[i]+signal_list_loaded[j]+noise)
                signal_a=signal_list_loaded[i]/np.max(signal_list_loaded[i])
                signal_b=signal_list_loaded[j]/np.max(signal_list_loaded[j])
                chirp_massa=chirp_mass_list_loaded[i]
                chirp_massb=chirp_mass_list_loaded[j]
                yield (mix_signal.reshape(-1, 4*4096, 1),signal_a.reshape(-1, 4*4096, 1),
                       signal_b.reshape(-1, 4*4096, 1),chirp_massa,chirp_massb)
# test_data=load_data_for_chirp_mass_test()
# for i,(mix_signal,signala,signalb,massa,massb) in enumerate(test_data):
#     plt.plot(mix_signal.reshape(16384))
#     plt.plot(signala.reshape(16384))
#     plt.plot(signalb.reshape(16384))
#     print(massa)
#     print(massb)
#     plt.show()
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
        test_data = load_data_for_chirp_mass_test()
        overlap_a={}
        overlap_b={}
        with torch.no_grad():
            for i,(egs,signala,signalb,massa,massb) in enumerate(test_data):
                start_time=time.time()
                # y0 =torch.from_numpy(y0)
                # noise = torch.from_numpy(noise)
                # signal1 = torch.from_numpy(signal1)
                # signal2 = torch.from_numpy(signal2)
                signala=torch.from_numpy(signala)
                signalb=torch.from_numpy(signalb)
                # if torch.argmax(y1)>=torch.argmax(y2):
                #    temp = y1.clone()
                #    y1.copy_(y2)
                #    y2.copy_(temp)
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

                # for i in range(0, len(spks), 2):
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
                    # if torch.argmax(s1) >= torch.argmax(s2):
                    #     temp = s1.clone()
                    #     s1.copy_(s2)
                    #     s2.copy_(temp)
                print('massa：',massa)
                print('massb：',massb)
                overlap1=overlap(s1.cpu().numpy(), signala.cpu().numpy())
                overlap2=overlap(-s2.cpu().numpy(), signalb.cpu().numpy())
                if overlap1<=0.65 and overlap2<=0.65:
                   overlap1 = overlap(-s2.cpu().numpy(), signala.cpu().numpy())
                   overlap2 = overlap(s1.cpu().numpy(), signalb.cpu().numpy())
                overlap_a[(massa,massb)]=overlap1
                overlap_b[(massa,massb)]=overlap2
        with open('overlap_a1.pkl','wb') as f:
            pickle.dump(overlap_a,f)
        with open('overlap_b1.pkl', 'wb') as f:
            pickle.dump(overlap_b, f)

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