import sys
sys.path.append('../')

from utils.util import check_parameters
import time
import logging
from logger.set_logger import setup_logger
from model.loss import Loss
import torch
import os
import matplotlib.pyplot as plt
from torch.nn.parallel import data_parallel
# from load_data_saparate import get_train_batch_iter
# from load_data_saparate import get_train_batch_iter_shuffle_epoch
from denoise_to_saparate_data import get_saparate_data_train
from  denoise_to_saparate_data import get_saparate_data_val

class Trainer(object):
    def __init__(self,  Conv_Tasnet,  optimizer, scheduler, opt):
        super(Trainer).__init__()
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
            self.convtasnet = Conv_Tasnet.to(self.device)
            self.logger.info(
                'Loading Conv-TasNet parameters: {:.3f} Mb'.format(check_parameters(self.convtasnet)))
        else:
            self.logger.info('Load CPU ...........')
            self.device = torch.device('cpu')
            self.convtasnet = Conv_Tasnet.to(self.device)
            self.logger.info(
                'Loading Conv-TasNet parameters: {:.3f} Mb'.format(check_parameters(self.convtasnet)))

        if opt['resume']['state']:
            ckp = torch.load(opt['resume']['path'], map_location='cpu')
            self.cur_epoch = ckp['epoch']
            self.logger.info("Resume from checkpoint {}: epoch {:.3f}".format(
                opt['resume']['path'], self.cur_epoch))
            self.convtasnet = Conv_Tasnet.load_state_dict(
                ckp['model_state_dict']).to(self.device)
            self.optimizer = optimizer.load_state_dict(ckp['optim_state_dict'])
        else:
            self.convtasnet = Conv_Tasnet.to(self.device)
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
        self.convtasnet.train()
        total_loss = 0.0
        num_index = 1
        start_time = time.time()
        # train_datapath = 'H://train'
        # val_datapath = 'H://val'
        # sample_length = 4
        # snr_low_list = [30, 26, 22, 17, 12, 12, 17, 22, 26]
        # snr_high_list = [26, 22, 17, 12, 8, 8, 12, 17, 22]
        # # snr_list=[50,45,40,35]
        # peak_range = (0.5, 0.95)
        # freq = 2048
        # snr_change_time = 2000
        batch_size = 16
        total_iteration = 3 * 100000 / batch_size
        # train_dataloader = get_train_batch_iter_shuffle_epoch(batch_size=batch_size, sample_length=sample_length,
        #                                       snr_low_list=snr_low_list, snr_high_list=snr_high_list,
        #                                       peak_range=peak_range, freq=freq, snr_change_time=snr_change_time,
        #                                       datapath=train_datapath)
        train_dataloader = get_saparate_data_train()
        for mix, sig1,sig2 in train_dataloader:
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
                model = torch.nn.DataParallel(self.convtasnet)

                out = model(mix)
                #out = self.convtasnet(mix)
            else:
                out = self.convtasnet(mix)

            # if torch.argmax(out[0])<=torch.argmax(out[1]):
            #     out[0],out[1]=out[1],out[0]
            # # print(out[0].shape)
            # l = Loss(sig1, sig2,out[0], out[1])
            l=Loss(out,sig1,sig2)
            epoch_loss = l
            total_loss += epoch_loss.item()
            epoch_loss.backward()

            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.convtasnet.parameters(), self.clip_norm)

            self.optimizer.step()
            if num_index % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>'.format(
                    epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss/num_index)
                self.logger.info(message)
            if num_index>=total_iteration:
                break
            num_index += 1
        end_time = time.time()
        total_loss = total_loss/num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, (end_time-start_time)/60)
        self.logger.info(message)
        return total_loss

    def validation(self, epoch):
        self.logger.info(
            'Start Validation from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.convtasnet.eval()
        total1_iteration = 20000 / 16
        num_index = 1
        total_loss = 0.0
        start_time = time.time()
        # val_datapath = 'H://val'
        # sample_length = 4
        # snr_low_list = [30, 26, 22, 17, 12, 12, 17, 22, 26]
        # snr_high_list = [26, 22, 17, 12, 8, 8, 12, 17, 22]
        # # snr_list=[50,45,40,35]
        # peak_range = (0.5, 0.95)
        # freq = 2048
        # snr_change_time = 2000
        # batch_size = 16
        # val_dataloader = get_train_batch_iter(batch_size=batch_size, sample_length=sample_length,
        #                                       snr_low_list=snr_low_list, snr_high_list=snr_high_list,
        #                                       peak_range=peak_range, freq=freq, snr_change_time=snr_change_time,
        #                                       datapath=val_datapath)
        val_dataloader = get_saparate_data_val()
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
                    #model = torch.nn.DataParallel(self.convtasnet)
                    #out = model(mix)
                    out = torch.nn.parallel.data_parallel(self.convtasnet,mix,device_ids=self.gpuid)
                    # out = self.convtasnet(mix)
                else:
                    out = self.convtasnet(mix)
                # if torch.argmax(out[0]) <= torch.argmax(out[1]):
                #     out[0], out[1] = out[1], out[0]
                #     # print(out[0].shape)
                # l = Loss(sig1, sig2, out[0], out[1])
                # if max(out[0])>max(out[1]):
                #     # out[0], out[1] = out[1], out[0]
                #     print('信号1')
                # else:
                #     print('信号2')
                # print(out[0].shape)
                # l = Loss(sig1, sig2,out[0], out[1])
                l = Loss(out, sig1, sig2)
                epoch_loss = l
                total_loss += epoch_loss.item()
                if num_index % self.print_freq == 0:
                    message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>'.format(
                        epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss/num_index)
                    self.logger.info(message)
                if num_index >= total1_iteration:
                    break
                num_index += 1
        print(num_index)
        end_time = time.time()
        total_loss = total_loss/num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, (end_time-start_time)/60)
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
        #plt.xticks(l, lx)
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
