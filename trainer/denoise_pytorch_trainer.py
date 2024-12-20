import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('../')
# import tensorflow as tf
from utils.util import check_parameters
import time
import logging
from logger.set_logger import setup_logger
from model.loss import Loss1
import torch
import os
import matplotlib.pyplot as plt
from torch.nn.parallel import data_parallel
from load_data_zwg import get_train_batch_iter
from load_data_zwg import get_train_batch_iter_shuffle_epoch

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv1d(1, 128, kernel_size=128, padding='same')
        self.conv2 = nn.Conv1d(1, 128, kernel_size=64, padding='same')
        self.conv3 = nn.Conv1d(1, 128, kernel_size=32, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        # self.conv1_1  = nn.Conv1d(384,64,kernel_size=64,padding='same')
        self.conv4 = nn.Conv1d(384, 64, kernel_size=64, padding='same')
        self.conv5 = nn.Conv1d(384, 64, kernel_size=32, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv6 = nn.Conv1d(128, 32, kernel_size=32, padding='same')
        self.conv7 = nn.Conv1d(128, 32, kernel_size=16, padding='same')
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.up1 = nn.Upsample(scale_factor=2)
        self.conv8 = nn.Conv1d(64, 32, kernel_size=16, padding='same')
        self.conv9 = nn.Conv1d(64, 32, kernel_size=32, padding='same')

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv10 = nn.Conv1d(192, 64, kernel_size=32, padding='same')
        self.conv11 = nn.Conv1d(64, 64, kernel_size=64, padding='same')

        self.up3 = nn.Upsample(scale_factor=4)
        self.conv12 = nn.Conv1d(512, 128, kernel_size=32, padding='same')
        self.conv13 = nn.Conv1d(128, 128, kernel_size=64, padding='same')
        self.conv14 = nn.Conv1d(128, 128, kernel_size=128, padding='same')

        self.conv15 = nn.Conv1d(384, 64, kernel_size=32, padding='same')
        self.conv16 = nn.Conv1d(64, 32, kernel_size=16, padding='same')
        self.conv17 = nn.Conv1d(32, 1, kernel_size=16, padding='same')

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = x.squeeze(-1)
        # print(x.shape)
        x = x.to(torch.float32)

        conv1_out = F.elu(self.conv1(x))
        conv2_out = F.elu(self.conv2(x))
        conv3_out = F.elu(self.conv3(x))
        merge1 = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)

        # print(merge1.shape)
        pool1_out = self.pool1(merge1)
        # print(pool1_out.shape)

        conv4_out = F.elu(self.conv4(pool1_out))
        conv5_out = F.elu(self.conv5(pool1_out))
        merge2 = torch.cat([conv4_out, conv5_out], dim=1)
        pool2_out = self.pool2(merge2)

        conv6_out = F.elu(self.conv6(pool2_out))
        conv7_out = F.elu(self.conv7(pool2_out))
        merge3 = torch.cat([conv6_out, conv7_out], dim=1)
        pool3_out = self.pool3(merge3)

        up1_out = self.up1(pool3_out)
        conv8_out = F.elu(self.conv8(up1_out))
        conv9_out = F.elu(self.conv9(up1_out))
        merge4 = torch.cat([conv8_out, conv9_out], dim=1)

        up2_out = self.up2(merge4)
        m2= torch.cat([up2_out, merge2], dim=1)
        conv10_out = F.elu(self.conv10(m2))
        conv11_out = F.elu(self.conv11(up2_out))
        merge5=torch.cat([conv10_out, conv11_out], dim=1)

        up3_out = self.up3(merge5)
        merge6 = torch.cat([up3_out, merge1], dim=1)
        conv12_out = F.elu(self.conv12(merge6))
        conv13_out = F.elu(self.conv13(up3_out))
        conv14_out = F.elu(self.conv14(up3_out))

        merge7 = torch.cat([conv12_out, conv13_out, conv14_out], dim=1)
        conv15_out = F.elu(self.conv15(merge7))
        conv16_out = F.elu(self.conv16(conv15_out))
        out = torch.tanh(self.conv17(conv16_out))
        # out = self.conv17(conv16_out)
        out=torch.squeeze(out,dim=1)
        return out
class Trainer(object):
    def __init__(self,MyModel, optimizer, scheduler, opt):
        super(Trainer).__init__()
        # self.train_dataloader = train_dataloader
        # self.val_dataloader = val_dataloader
        self.scheduler = scheduler
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
            self.convtasnet = MyModel.to(self.device)
            self.logger.info(
                'Loading Conv-TasNet parameters: {:.3f} Mb'.format(check_parameters(self.convtasnet)))
        else:
            self.logger.info('Load CPU ...........')
            self.device = torch.device('cpu')
            self.convtasnet = MyModel.to(self.device)
            self.logger.info(
                'Loading Mymodel parameters: {:.3f} Mb'.format(check_parameters(self.convtasnet)))

        if opt['resume']['state']:
            ckp = torch.load(os.path.join(
                opt['resume']['path'], 'best.pt'), map_location='cpu')
            self.cur_epoch = ckp['epoch']
            self.logger.info("Resume from checkpoint {}: epoch {:.3f}".format(
                opt['resume']['path'], self.cur_epoch))
            MyModel.load_state_dict(
                ckp['model_state_dict'])
            self.dualrnn = MyModel.to(self.device)
            optimizer.load_state_dict(ckp['optim_state_dict'])
            self.optimizer = optimizer
            lr = self.optimizer.param_groups[0]['lr']
            # self.adjust_learning_rate(self.optimizer, lr * 0.5)
        else:
            self.dualrnn = MyModel.to(self.device)
            self.optimizer = optimizer
        # if opt['resume']['state']:
        #     ckp = torch.load(opt['resume']['path'], map_location='cpu')
        #     self.cur_epoch = ckp['epoch']
        #     self.logger.info("Resume from checkpoint {}: epoch {:.3f}".format(
        #         opt['resume']['path'], self.cur_epoch))
        #     self.convtasnet = MyModel.load_state_dict(
        #         ckp['model_state_dict']).to(self.device)
        #     self.optimizer = optimizer.load_state_dict(ckp['optim_state_dict'])
        # else:
        #     self.convtasnet = MyModel.to(self.device)
        #     self.optimizer = optimizer

        # if opt['optim']['clip_norm']:
        #     self.clip_norm = opt['optim']['clip_norm']
        #     self.logger.info(
        #         "Gradient clipping by {}, default L2".format(self.clip_norm))
        # else:
        #     self.clip_norm = 0

    def train(self, epoch):
        self.logger.info(
            'Start training from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.convtasnet.train()
        total_loss = 0.0
        num_index = 1
        start_time = time.time()
        train_datapath = 'H://10-80_10-80'
        val_datapath = 'H://val'
        sample_length = 4
        snr_low_list = [30, 26, 22, 17, 12, 12, 17, 22, 26]
        snr_high_list = [26, 22, 17, 12, 8, 8, 12, 17, 22]
        # snr_list=[50,45,40,35]
        peak_range = (0.5, 0.95)
        freq = 4096
        snr_change_time = 2000
        batch_size = 16
        total_iteration = 3 * 100000 / batch_size
        train_dataloader = get_train_batch_iter_shuffle_epoch(batch_size=batch_size, sample_length=sample_length,
                                                              snr_low_list=snr_low_list, snr_high_list=snr_high_list,
                                                              peak_range=peak_range, freq=freq,
                                                              snr_change_time=snr_change_time,
                                                              datapath=train_datapath)
        for mix, sig in train_dataloader:
            mix = torch.from_numpy(mix)
            mix = mix.to(self.device).float()
            # ref = [ref[i].to(self.device) for i in range(self.num_spks)]
            sig = torch.from_numpy(sig)
            sig = sig.to(self.device).squeeze(-1).float()
            # sig = torch.unsqueeze(sig, dim=1)
            # print(sig.shape)
            self.optimizer.zero_grad()

            if self.gpuid:
                model = torch.nn.DataParallel(self.convtasnet)

                out = model(mix)
                # out = self.convtasnet(mix)
            else:
                out = self.convtasnet(mix)

            # if torch.argmax(out[0]) <= torch.argmax(out[1]):
            #     out[0], out[1] = out[1], out[0]
            # # print(out[0].shape)
            # l = Loss(sig1, sig2, out[0], out[1])
            l=Loss1(sig,out)
            epoch_loss = l
            total_loss += epoch_loss.item()
            epoch_loss.backward()

            # if self.clip_norm:
            #     torch.nn.utils.clip_grad_norm_(
            #         self.convtasnet.parameters(), self.clip_norm)

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
        self.convtasnet.eval()
        total1_iteration=20000/16
        num_index = 1
        total_loss = 0.0
        start_time = time.time()
        val_datapath = 'H://val'
        sample_length = 4
        snr_low_list = [30, 26, 22, 17, 12, 12, 17, 22, 26]
        snr_high_list = [26, 22, 17, 12, 8, 8, 12, 17, 22]
        # snr_list=[50,45,40,35]
        peak_range = (0.5, 0.95)
        freq = 4096
        snr_change_time = 2000
        batch_size = 16
        val_dataloader = get_train_batch_iter(batch_size=batch_size, sample_length=sample_length,
                                              snr_low_list=snr_low_list, snr_high_list=snr_high_list,
                                              peak_range=peak_range, freq=freq, snr_change_time=snr_change_time,
                                              datapath=val_datapath)
        with torch.no_grad():
            for mix, sig in val_dataloader:
                mix = torch.from_numpy(mix)
                mix = mix.to(self.device).float()
                # ref = [ref[i].to(self.device) for i in range(self.num_spks)]
                sig = torch.from_numpy(sig)
                sig = sig.to(self.device).squeeze(-1).float()
                # sig = torch.unsqueeze(sig, dim=1)
                # print(sig.shape)
                self.optimizer.zero_grad()

                if self.gpuid:
                    model = torch.nn.DataParallel(self.convtasnet)

                    out = model(mix)
                    # out = self.convtasnet(mix)
                else:
                    out = self.convtasnet(mix)

                # if torch.argmax(out[0]) <= torch.argmax(out[1]):
                #     out[0], out[1] = out[1], out[0]
                # # print(out[0].shape)
                # l = Loss(sig1, sig2, out[0], out[1])
                l=Loss1(sig,out)
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

    def run(self):
        train_loss = []
        val_loss = []
        with torch.cuda.device(self.gpuid[0]):
            self.save_checkpoint(self.cur_epoch, best=False)
            v_loss = self.validation(self.cur_epoch)
            best_loss = v_loss

            self.logger.info("Starting epoch from {:d}, loss = {:.4f}".format(
                self.cur_epoch, best_loss))
            no_improve = 0
            # starting training part
            while self.cur_epoch < self.total_epoch:
                self.cur_epoch += 1
                t_loss = self.train(self.cur_epoch)
                v_loss = self.validation(self.cur_epoch)

                train_loss.append(t_loss)
                val_loss.append(v_loss)

                # schedule here
                self.scheduler.step(v_loss)

                if v_loss >= best_loss:
                    no_improve += 1
                    self.logger.info(
                        'No improvement, Best Loss: {:.4f}'.format(best_loss))
                else:
                    best_loss = v_loss
                    no_improve = 0
                    self.save_checkpoint(self.cur_epoch, best=True)
                    self.logger.info('Epoch: {:d}, Now Best Loss Change: {:.4f}'.format(
                        self.cur_epoch, best_loss))

                if no_improve == self.early_stop:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_improve))
                    break
            self.save_checkpoint(self.cur_epoch, best=False)
            self.logger.info("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, self.total_epoch))

        # draw loss image
        plt.title("Loss of train and test")
        x = [i for i in range(self.cur_epoch)]
        plt.plot(x, train_loss, 'b-', label=u'train_loss', linewidth=0.8)
        plt.plot(x, val_loss, 'c-', label=u'val_loss', linewidth=0.8)
        plt.legend()
        # plt.xticks(l, lx)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('loss.png')

    def save_checkpoint(self, epoch, best=True):
        '''
           save model
           best: the best model
        '''
        os.makedirs(os.path.join(self.checkpoint, self.name), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.convtasnet.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        },
            os.path.join(self.checkpoint, self.name, '{0}.pt'.format('best' if best else 'last')))
