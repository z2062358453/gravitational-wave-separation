import sys
sys.path.append('../')

from utils.util import check_parameters
import time
import logging
from logger.set_logger import setup_logger
from model.loss import Loss
from model.loss import combined_loss
import torch
import os
import matplotlib.pyplot as plt
from torch.nn.parallel import data_parallel
# from load_data_saparate import get_train_batch_iter
# from load_data_saparate import get_train_batch_iter_shuffle_epoch
from denoise_to_saparate_data import get_saparate_data_train
from  denoise_to_saparate_data import get_saparate_data_val


class Trainer(object):
    def __init__(self, Dual_RNN,  optimizer, scheduler, opt):
        super(Trainer).__init__()
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
            self.dualrnn = Dual_RNN.to(self.device)
            self.logger.info(
                'Loading Dual-Path-RNN parameters: {:.3f} Mb'.format(check_parameters(self.dualrnn)))
        else:
            self.logger.info('Load CPU ...........')
            self.device = torch.device('cpu')
            self.dualrnn = Dual_RNN.to(self.device)
            self.logger.info(
                'Loading Dual-Path-RNN parameters: {:.3f} Mb'.format(check_parameters(self.dualrnn)))

        if opt['resume']['state']:
            ckp = torch.load(os.path.join(
                opt['resume']['path'], 'best_rnn_mse1.pt'), map_location='cpu')
            self.cur_epoch = ckp['epoch']
            self.logger.info("Resume from checkpoint {}: epoch {:.3f}".format(
                opt['resume']['path'], self.cur_epoch))
            Dual_RNN.load_state_dict(
                ckp['model_state_dict'])
            self.dualrnn = Dual_RNN.to(self.device)
            optimizer.load_state_dict(ckp['optim_state_dict'])
            self.optimizer = optimizer
            lr = self.optimizer.param_groups[0]['lr']
            # self.adjust_learning_rate(self.optimizer, lr*0.5)
        else:
            self.dualrnn = Dual_RNN.to(self.device)
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
        self.dualrnn.train()
        batch_size=4
        total_iteration = 100000 / batch_size
        total_loss = 0.0
        num_index = 1
        start_time = time.time()

        train_dataloader = get_saparate_data_train()
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
                out = torch.nn.parallel.data_parallel(self.dualrnn,mix,device_ids=self.gpuid)
                #out = self.dualrnn(mix)
            else:
                out = self.dualrnn(mix)
            if torch.argmax(sig1) <= torch.argmax(sig2):
                    temp = sig1.clone()
                    sig1.copy_(sig2)
                    sig2.copy_(temp)
            # if torch.argmax(out[0]) >= torch.argmax(out[1]):
            #     temp = out[0].clone()
            #     out[0].copy_(out[1])
            #     out[1].copy_(temp)
            # l=Loss(out,sig1,sig2)
            l = combined_loss(sig1, sig2, out[0], out[1])
            epoch_loss = l
            total_loss += epoch_loss.item()
            epoch_loss.backward()

            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.dualrnn.parameters(), self.clip_norm)

            self.optimizer.step()
            if num_index % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>'.format(
                    epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss/num_index)
                self.logger.info(message)
            if num_index >= total_iteration:
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
        self.dualrnn.eval()
        total1_iteration = 10000 / 4
        num_index = 1
        total_loss = 0.0
        start_time = time.time()
        val_dataloader = get_saparate_data_val()
        with torch.no_grad():
            for mix, sig1,sig2 in val_dataloader:
                mix = torch.from_numpy(mix)
                mix = mix.to(self.device)
                # ref = [ref[i].to(self.device) for i in range(self.num_spks)]
                sig1 = torch.from_numpy(sig1)
                sig2 = torch.from_numpy(sig2)
                sig1 = sig1.to(self.device).squeeze(-1)
                sig2 = sig2.to(self.device).squeeze(-1)
                self.optimizer.zero_grad()

                if self.gpuid:
                    #model = torch.nn.DataParallel(self.dualrnn)
                    #out = model(mix)
                    out = torch.nn.parallel.data_parallel(self.dualrnn,mix,device_ids=self.gpuid)
                else:
                    out = self.dualrnn(mix)
                if torch.argmax(sig1) <= torch.argmax(sig2):
                    temp = sig1.clone()
                    sig1.copy_(sig2)
                    sig2.copy_(temp)
                # if torch.argmax(out[0]) >= torch.argmax(out[1]):
                #     temp = out[0].clone()
                #     out[0].copy_(out[1])
                #     out[1].copy_(temp)
                l = combined_loss(sig1, sig2, out[0], out[1])
                # l = Loss(out, sig1, sig2)
                epoch_loss = l
                total_loss += epoch_loss.item()
                if num_index % self.print_freq == 0:
                    message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>'.format(
                        epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss/num_index)
                    self.logger.info(message)
                if num_index >= total1_iteration:
                    break
                num_index += 1
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
            'model_state_dict': self.dualrnn.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        },
            os.path.join(self.checkpoint, self.name, '{0}.pt'.format('best_rnn_mse1' if best else 'last')))
