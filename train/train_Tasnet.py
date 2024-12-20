import sys
sys.path.append('./')

from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.data import DataLoader as Loader
# from data_loader.Dataset import Datasets
from model import model
from logger import set_logger
import logging
from config import option
import argparse
import torch
from trainer import trainer_Tasnet
# from load_data_saparate import get_train_batch_iter
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
# def make_dataloader():
#     # make train's dataloader
#     #
#     # train_dataset = Datasets(
#     #     opt['datasets']['train']['dataroot_mix'],
#     #     [opt['datasets']['train']['dataroot_targets'][0],
#     #      opt['datasets']['train']['dataroot_targets'][1]],
#     #     **opt['datasets']['audio_setting'])
#     # train_dataloader = Loader(train_dataset,
#     #                           batch_size=opt['datasets']['dataloader_setting']['batch_size'],
#     #                           num_workers=opt['datasets']['dataloader_setting']['num_workers'],
#     #                           shuffle=opt['datasets']['dataloader_setting']['shuffle'])
#     #
#     # # make validation dataloader
#     #
#     # val_dataset = Datasets(
#     #     opt['datasets']['val']['dataroot_mix'],
#     #     [opt['datasets']['val']['dataroot_targets'][0],
#     #      opt['datasets']['val']['dataroot_targets'][1]],
#     #     **opt['datasets']['audio_setting'])
#     # val_dataloader = Loader(val_dataset,
#     #                         batch_size=opt['datasets']['dataloader_setting']['batch_size'],
#     #                         num_workers=opt['datasets']['dataloader_setting']['num_workers'],
#     #                         shuffle=opt['datasets']['dataloader_setting']['shuffle'])
#     #
#     train_datapath = 'H://train'
#     val_datapath = 'H://val'
#     sample_length = 4
#     snr_low_list = [30, 26, 22, 17, 12, 12, 17, 22, 26]
#     snr_high_list = [26, 22, 17, 12, 8, 8, 12, 17, 22]
#     # snr_list=[50,45,40,35]
#     peak_range = (0.5, 0.95)
#     freq = 2048
#     snr_change_time = 2000
#     batch_size = 16
#     train_dataloader=get_train_batch_iter(batch_size=batch_size,sample_length=sample_length,
#                                           snr_low_list=snr_low_list,snr_high_list=snr_high_list,
#                                           peak_range=peak_range,freq=freq, snr_change_time=snr_change_time,
#                                           datapath=train_datapath)
#     val_dataloader = get_train_batch_iter(batch_size=batch_size, sample_length=sample_length,
#                                           snr_low_list=snr_low_list, snr_high_list=snr_high_list,
#                                           peak_range=peak_range, freq=freq, snr_change_time=snr_change_time,
#                                           datapath=val_datapath)
#     return train_dataloader, val_dataloader


def make_optimizer(params, opt):
    optimizer = getattr(torch.optim, opt['optim']['name'])
    if opt['optim']['name'] == 'Adam':
        optimizer = optimizer(
            params, lr=opt['optim']['lr'], weight_decay=opt['optim']['weight_decay'])
    else:
        optimizer = optimizer(params, lr=opt['optim']['lr'], weight_decay=opt['optim']
                              ['weight_decay'], momentum=opt['optim']['momentum'])

    return optimizer


def train():
    parser = argparse.ArgumentParser(
        description='Parameters for training Conv-TasNet')
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = option.parse(args.opt)
    set_logger.setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
    logger = logging.getLogger(opt['logger']['name'])
    # build model
    logger.info("Building the model of Conv-Tasnet")
    Conv_Tasnet = model.Conv_TasNet(**opt['Conv_Tasnet'])
    # build optimizer
    logger.info("Building the optimizer of Conv-Tasnet")
    optimizer = make_optimizer(Conv_Tasnet.parameters(), opt)
    # build dataloader
    logger.info('Building the dataloader of Conv-Tasnet')
    # build scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=opt['scheduler']['factor'],
        patience=opt['scheduler']['patience'],
        verbose=True, min_lr=opt['scheduler']['min_lr'])
    
    # build trainer
    logger.info('Building the Trainer of Conv-Tasnet')
    trainer = trainer_Tasnet.Trainer(Conv_Tasnet, optimizer, scheduler, opt)
    trainer.run()


if __name__ == "__main__":
    train()
