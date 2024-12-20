import sys

sys.path.append('/')
import denoise_pytorch_trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from logger import set_logger
import logging
from config import option
import argparse
import torch
import end_to_end_denoise_saparate_trainer
import random
import numpy as np
from model.model import Conv_TasNet
from config.option1 import parse
from torch.autograd import  Variable
from model.model_rnn import Dual_RNN_model
class data_sample():
    def __init__(self, noiseE1, noiseE2, noiseE3, mass1_list, mass2_list, spin1z_list, spin2z_list,
                 right_ascension_list,
                 declination_list, signal_E1_list, signal_E2_list, signal_E3_list, snr_E1_list, snr_E2_list,
                 snr_E3_list):
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
        print('mass1=' + str(mass1_list))
        print('mass2=' + str(mass2_list))
        print('spin1z=' + str(spin1z_list))
        print('spin2z=' + str(spin2z_list))
        print('right_ascension=' + str(right_ascension_list))
        print('declination=' + str(declination_list))

    def help(self):
        print('noiseE1, noiseE2, noiseE3 are 128 s length noise of E1, E2 and E3')
        print('signal_E1_list, signal_E2_list and signal_E3_list are signals, and each have 1 samples')
        print('mass1_list, mass2_list are masses, and each have 1 masses')
        print('right_ascension_list and declination_list are the directions of the source of the signal')
# 加载第一个模型
model_denoise = denoise_pytorch_trainer.MyModel()
model_path = './checkpoint_mse/MyModel/best.pt'
dict1 = torch.load(model_path, map_location='cuda:0')
model_denoise.load_state_dict(dict1["model_state_dict"])
model_denoise = model_denoise.cuda()


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
dict3 = torch.load(model_path, map_location='cpu')
Dual_Path_RNN.load_state_dict(dict3["model_state_dict"])
Dual_Path_RNN = Dual_Path_RNN.cuda()
# 将两个模型组合成一个模型
def make_optimizer(params, opt):
    optimizer = getattr(torch.optim, opt['optim']['name'])
    if opt['optim']['name'] == 'Adam':
        # optimizer = optimizer(
        #     params, lr=opt['optim']['lr'], weight_decay=opt['optim']['weight_decay'])
        optimizer = optimizer(
            params, lr=opt['optim']['lr'])
    else:
        optimizer = optimizer(params, lr=opt['optim']['lr'], weight_decay=opt['optim']
        ['weight_decay'], momentum=opt['optim']['momentum'])

    return optimizer


def train():
    parser = argparse.ArgumentParser(
        description='Parameters for training end_to_end_model')
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = option.parse(args.opt)
    set_logger.setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
    logger = logging.getLogger(opt['logger']['name'])
    # build model
    logger.info("Building the model of end_to_end_model")
    end_to_end_model = end_to_end_denoise_saparate_trainer.CombinedModel(model_denoise,Dual_Path_RNN)
    # build optimizer
    logger.info("Building the optimizer of end_to_end_model")
    optimizer = make_optimizer(end_to_end_model.parameters(), opt)
    # build dataloader
    logger.info('Building the dataloader of end_to_end_model')
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=opt['scheduler']['factor'],
        patience=opt['scheduler']['patience'],
        verbose=True, min_lr=opt['scheduler']['min_lr'])

    # build trainer
    logger.info('Building the Trainer of end_to_end_model')
    trainer = end_to_end_denoise_saparate_trainer.end_to_end_Trainer(end_to_end_model, optimizer, scheduler, opt)
    trainer.run()


if __name__ == "__main__":
    # seed=2022
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    train()
