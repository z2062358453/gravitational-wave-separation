import sys
sys.path.append('../')
from config.option import parse
from utils.util import check_parameters
import time
import logging
from logger.set_logger import setup_logger
from model.loss import Loss
import torch
import os
import matplotlib.pyplot as plt
from torch.nn.parallel import data_parallel
from load_data_saparate import get_train_batch_for_end2end
from load_data_saparate import get_val_batch_for_end2end
# from denoise_to_saparate_data import get_saparate_data_train
# from  denoise_to_saparate_data import get_saparate_data_val
from model.model_rnn import Dual_RNN_model
import denoise_pytorch_trainer
from model.model import Conv_TasNet
from config import option
import argparse
import torch
from torch import nn
from config.option import parse
import csv
# 加载第一个模型
model_denoise = denoise_pytorch_trainer.MyModel()
# model_path = 'E://Dual-Path-RNN-Pytorch//checkpoint_mse//MyModel//best.pt'
model_path = './checkpoint_mse/MyModel/best.pt'
dict1 = torch.load(model_path, map_location='cuda:0')
model_denoise.load_state_dict(dict1["model_state_dict"])
model_denoise = model_denoise.cuda()
# model_denoise = model_denoise
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
# model_path = 'E://Dual-Path-RNN-Pytorch//checkpoint//Dual_Path_RNN//best_rnn.pt'
model_path = './checkpoint/Dual_Path_RNN/best_rnn.pt'
dict3 = torch.load(model_path, map_location='cpu')
Dual_Path_RNN.load_state_dict(dict3["model_state_dict"])
Dual_Path_RNN = Dual_Path_RNN.cuda()
# Dual_Path_RNN = Dual_Path_RNN
# 将两个模型组合成一个模型
class CombinedModel(nn.Module):
    def __init__(self, model_denoise, Dual_Path_RNN):
        super(CombinedModel, self).__init__()
        self.model_denoise = model_denoise
        self.Dual_Path_RNN = Dual_Path_RNN

    def forward(self, x):
        # 对第一个模型进行前向传播
        x = self.model_denoise(x)
        # 对第二个模型进行前向传播
        x = self.Dual_Path_RNN(x)
        return x


class end_to_end_Trainer(object):
    def __init__(self, CombinedModel, optimizer, scheduler, opt):
        super(end_to_end_Trainer).__init__()
        # self.train_dataloader = train_dataloader
        # self.val_dataloader = val_dataloader
        self.scheduler = scheduler
        self.num_spks = opt['num_spks']
        self.cur_epoch = 0
        self.total_epoch = opt['train']['epoch']
        self.early_stop = opt['train']['early_stop']

        self.print_freq = opt['logger']['print_freq']
        # setup_logger(opt['logger']['name'], opt['logger']['path'],
        #             screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
        self.logger = logging.getLogger(opt['logger']['name'])
        self.checkpoint = opt['train']['path']
        self.name = opt['name']

        if opt['train']['gpuid']:
            self.logger.info('Load Nvida GPU .....')
            self.device = torch.device(
                'cuda:{}'.format(opt['train']['gpuid'][0]))
            self.gpuid = opt['train']['gpuid']
            self.end_to_end_model = CombinedModel.to(self.device)
            self.logger.info(
                'Loading Conv-TasNet parameters: {:.3f} Mb'.format(check_parameters(self.end_to_end_model)))
        else:
            self.logger.info('Load CPU ...........')
            self.device = torch.device('cpu')
            self.end_to_end_model = CombinedModel.to(self.device)
            self.logger.info(
                'Loading end_to_end_model parameters: {:.3f} Mb'.format(check_parameters(self.end_to_end_model)))

        if opt['resume']['state']:
            ckp = torch.load(opt['resume']['path'], map_location='cpu')
            self.cur_epoch = ckp['epoch']
            self.logger.info("Resume from checkpoint {}: epoch {:.3f}".format(
                opt['resume']['path'], self.cur_epoch))
            self.end_to_end_model = CombinedModel.load_state_dict(
                ckp['model_state_dict']).to(self.device)
            self.optimizer = optimizer.load_state_dict(ckp['optim_state_dict'])
        else:
            self.end_to_end_model = CombinedModel.to(self.device)
            self.optimizer = optimizer

        if opt['optim']['clip_norm']:
            self.clip_norm = opt['optim']['clip_norm']
            self.logger.info(
                "Gradient clipping by {}, default L2".format(self.clip_norm))
        else:
            self.clip_norm = 0

    def train(self, epoch):
        self.logger.info(
            'Start training from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.end_to_end_model.train()
        total_loss = 0.0
        num_index = 1
        start_time = time.time()
        batch_size = 4
        total_iteration = 25000
        train_datapath = 'H://10-80_10-80'
        val_datapath = 'H://val'
        sample_length = 4
        snr_low_list = [30, 26, 22, 17, 12, 12, 17, 22, 26]
        snr_high_list = [26, 22, 17, 12, 8, 8, 12, 17, 22]
        peak_range = (0.5, 0.95)
        freq = 4096
        snr_change_time = 2000
        #     batch_size = 16
        train_dataloader = get_train_batch_for_end2end(batch_size=batch_size, sample_length=sample_length,
                                              snr_low_list=snr_low_list, snr_high_list=snr_high_list,
                                              peak_range=peak_range, freq=freq, snr_change_time=snr_change_time,
                                              datapath=train_datapath)
        # train_dataloader = get_saparate_data_train()
        for mix, sig1, sig2 in train_dataloader:
            mix = torch.from_numpy(mix)
            mix = mix.to(self.device).float()
            # ref = [ref[i].to(self.device) for i in range(self.num_spks)]
            sig1 = torch.from_numpy(sig1)
            sig2 = torch.from_numpy(sig2)
            sig1 = sig1.to(self.device).squeeze(-1).float()
            sig2 = sig2.to(self.device).squeeze(-1).float()
            # print(sig1.shape)
            self.optimizer.zero_grad()

            if self.gpuid:
                model = torch.nn.DataParallel(self.end_to_end_model)

                out = model(mix)
                # out = self.convtasnet(mix)
            else:
                out = self.end_to_end_model(mix)

            # if torch.argmax(out[0])<=torch.argmax(out[1]):
            #     out[0],out[1]=out[1],out[0]
            # # print(out[0].shape)
            # l = Loss(sig1, sig2,out[0], out[1])
            l = Loss(out, sig1, sig2)
            epoch_loss = l
            total_loss += epoch_loss.item()
            epoch_loss.backward()

            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.end_to_end_model.parameters(), self.clip_norm)

            self.optimizer.step()
            if num_index % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>'.format(
                    epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss / num_index)
                self.logger.info(message)
            if num_index >= total_iteration:
                break
            num_index += 1
        end_time = time.time()
        total_loss = total_loss / num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, (end_time - start_time) / 60)
        self.logger.info(message)
        return total_loss

    def validation(self, epoch):
        self.logger.info(
            'Start Validation from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.end_to_end_model.eval()
        total1_iteration = 2500
        num_index = 1
        total_loss = 0.0
        start_time = time.time()
        train_datapath = 'H://train'
        val_datapath = 'H://val'
        sample_length = 4
        snr_low_list = [30, 26, 22, 17, 12, 12, 17, 22, 26]
        snr_high_list = [26, 22, 17, 12, 8, 8, 12, 17, 22]
        peak_range = (0.5, 0.95)
        freq = 4096
        snr_change_time = 2000
        batch_size=4
        val_dataloader = get_val_batch_for_end2end(batch_size=batch_size, sample_length=sample_length,
                                              snr_low_list=snr_low_list, snr_high_list=snr_high_list,
                                              peak_range=peak_range, freq=freq, snr_change_time=snr_change_time,
                                              datapath=val_datapath)
        # val_dataloader = get_saparate_data_val()
        with torch.no_grad():
            for mix, sig1, sig2 in val_dataloader:
                mix = torch.from_numpy(mix)
                mix = mix.to(self.device)
                # ref = [ref[i].to(self.device) for i in range(self.num_spks)]
                sig1 = torch.from_numpy(sig1)
                sig2 = torch.from_numpy(sig2)
                sig1 = sig1.to(self.device).squeeze(-1)
                sig2 = sig2.to(self.device).squeeze(-1)
                self.optimizer.zero_grad()
                # print(sig1.shape)
                if self.gpuid:
                    # model = torch.nn.DataParallel(self.convtasnet)
                    # out = model(mix)
                    out = torch.nn.parallel.data_parallel(self.end_to_end_model, mix, device_ids=self.gpuid)
                    # out = self.convtasnet(mix)
                else:
                    out = self.end_to_end_model(mix)
                # if torch.argmax(out[0]) <= torch.argmax(out[1]):
                #     out[0], out[1] = out[1], out[0]
                #     # print(out[0].shape)
                # l = Loss(sig1, sig2, out[0], out[1])
                l = Loss(out, sig1, sig2)
                epoch_loss = l
                total_loss += epoch_loss.item()
                if num_index % self.print_freq == 0:
                    message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>'.format(
                        epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss / num_index)
                    self.logger.info(message)
                if num_index >= total1_iteration:
                    break
                num_index += 1
        print(num_index)
        end_time = time.time()
        total_loss = total_loss / num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, (end_time - start_time) / 60)
        self.logger.info(message)
        return total_loss

    # def run(self):
    #     train_loss = []
    #     val_loss = []
    #     with torch.cuda.device(self.gpuid[0]):
    #         self.save_checkpoint(self.cur_epoch, best=False)
    #         v_loss = self.validation(self.cur_epoch)
    #         best_loss = v_loss
    #
    #         self.logger.info("Starting epoch from {:d}, loss = {:.4f}".format(
    #             self.cur_epoch, best_loss))
    #         no_improve = 0
    #         # starting training part
    #         while self.cur_epoch < self.total_epoch:
    #             self.cur_epoch += 1
    #             t_loss = self.train(self.cur_epoch)
    #             v_loss = self.validation(self.cur_epoch)
    #
    #             train_loss.append(t_loss)
    #             val_loss.append(v_loss)
    #
    #             # schedule here
    #             self.scheduler.step(v_loss)
    #
    #             if v_loss >= best_loss:
    #                 no_improve += 1
    #                 self.logger.info(
    #                     'No improvement, Best Loss: {:.4f}'.format(best_loss))
    #             else:
    #                 best_loss = v_loss
    #                 no_improve = 0
    #                 self.save_checkpoint(self.cur_epoch, best=True)
    #                 self.logger.info('Epoch: {:d}, Now Best Loss Change: {:.4f}'.format(
    #                     self.cur_epoch, best_loss))
    #
    #             if no_improve == self.early_stop:
    #                 self.logger.info(
    #                     "Stop training cause no impr for {:d} epochs".format(
    #                         no_improve))
    #                 break
    #         self.save_checkpoint(self.cur_epoch, best=False)
    #         self.logger.info("Training for {:d}/{:d} epoches done!".format(
    #             self.cur_epoch, self.total_epoch))
    #
    #     # draw loss image
    #     plt.title("Loss of train and test")
    #     x = [i for i in range(self.cur_epoch)]
    #     plt.plot(x, train_loss, 'b-', label=u'train_loss', linewidth=0.8)
    #     plt.plot(x, val_loss, 'c-', label=u'val_loss', linewidth=0.8)
    #     plt.legend()
    #     # plt.xticks(l, lx)
    #     plt.ylabel('loss')
    #     plt.xlabel('epoch')
    #     plt.savefig('loss.png')
    # def plot_loss(self,train_loss, val_loss, save_path_base='loss'):
    #     """绘制并保存损失图"""
    #     plt.figure()
    #     plt.title("Loss of train and val")
    #     x = [i for i in range(len(train_loss))]
    #     plt.plot(x, train_loss, 'b-', label='train_loss', linewidth=0.8)
    #     plt.plot(x, val_loss, 'c-', label='val_loss', linewidth=0.8)
    #     plt.legend()
    #     plt.ylabel('loss')
    #     plt.xlabel('epoch')
    #     plt.savefig(f"{save_path_base}.tif",format='tif',dpi=300)
    #     plt.savefig(f"{save_path_base}.pdf",format='pdf', dpi=300)
    #     plt.close()  # 关闭绘图，释放内存

    def plot_loss_from_file(self,filename='losses.csv', save_path='loss.tif'):
        """从文件读取损失值并绘制图像"""
        train_loss = []
        val_loss = []

        # 读取文件中的损失值
        with open(filename, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                train_loss.append(float(row[0]))
                val_loss.append(float(row[1]))

        # 绘制损失图
        plt.figure()
        plt.title("Loss of train and test")
        x = [i for i in range(len(train_loss))]
        plt.plot(x, train_loss, 'b-', label='train_loss', linewidth=0.8)
        plt.plot(x, val_loss, 'c-', label='val_loss', linewidth=0.8)
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epoch')

        # 保存为 PNG 格式
        plt.savefig(save_path)
        plt.close()

    def save_loss_to_file(self,train_loss, val_loss, filename='losses.csv'):
        """将每轮训练和验证损失保存到文件"""
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([train_loss, val_loss])

    def run(self):
        with torch.cuda.device(self.gpuid[0]):
            self.save_checkpoint(self.cur_epoch, best=False)
            v_loss = self.validation(self.cur_epoch)
            best_loss = v_loss

            self.logger.info("Starting epoch from {:d}, loss = {:.4f}".format(self.cur_epoch, best_loss))
            no_improve = 0
            # 开始训练过程
            while self.cur_epoch < self.total_epoch:
                self.cur_epoch += 1
                t_loss = self.train(self.cur_epoch)
                v_loss = self.validation(self.cur_epoch)

                # 实时保存每轮的训练损失和验证损失到文件
                self.save_loss_to_file(t_loss, v_loss, filename='losses.csv')

                # 更新学习率调度器
                self.scheduler.step(v_loss)

                if v_loss >= best_loss:
                    no_improve += 1
                    self.logger.info('No improvement, Best Loss: {:.4f}'.format(best_loss))
                else:
                    best_loss = v_loss
                    no_improve = 0
                    self.save_checkpoint(self.cur_epoch, best=True)
                    self.logger.info('Epoch: {:d}, Now Best Loss Change: {:.4f}'.format(self.cur_epoch, best_loss))

                if no_improve == self.early_stop:
                    self.logger.info("Stop training cause no impr for {:d} epochs".format(no_improve))
                    break

            self.save_checkpoint(self.cur_epoch, best=False)
            self.logger.info("Training for {:d}/{:d} epochs done!".format(self.cur_epoch, self.total_epoch))

    def save_checkpoint(self, epoch, best=True):
        '''
           save model
           best: the best model
        '''
        os.makedirs(os.path.join(self.checkpoint, self.name), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.end_to_end_model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        },
            os.path.join(self.checkpoint, self.name, '{0}.pt'.format('best_end2end_plot_loss' if best else 'last')))
# import matplotlib.pyplot as plt
# import csv

def plot_loss_from_file(filename='losses.csv', save_path='loss.tif'):
    """从文件读取损失值并绘制图像"""
    train_loss = []
    val_loss = []

    # 读取文件中的损失值
    with open(filename, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            train_loss.append(float(row[0]))
            val_loss.append(float(row[1]))

    # 绘制损失图
    plt.figure()
    plt.title("Loss of train and test")
    x = [i for i in range(len(train_loss))]
    plt.plot(x, train_loss, 'b-', label='train_loss', linewidth=0.8)
    plt.plot(x, val_loss, 'c-', label='val_loss', linewidth=0.8)
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')

    # 设置横坐标刻度间隔为 2，并且只显示整数刻度
    plt.xticks(range(0, len(train_loss), 2))  # 设置刻度从 0 开始，步长为 2

    # 保存为 PNG 格式
    # plt.savefig(save_path)
    plt.show()
    plt.close()

# plot_loss_from_file(filename='losses.csv', save_path='loss.pdf')