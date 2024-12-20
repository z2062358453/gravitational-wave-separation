import os
import torch
import argparse
import denoise_pytorch_trainer
from model.model import Conv_TasNet
import argparse
import torch
from config.option import parse
from torch.nn.parallel import data_parallel
from model.model import Conv_TasNet
from logger.set_logger import setup_logger
import logging
from config.option import parse
import matplotlib.pyplot as plt
# from load_data_saparate import get_train_batch_iter
from denoise_to_saparate_data import get_saparate_data_test
import matplotlib
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

class Separation():
    def __init__(self, yaml_path, model, gpuid):
        super(Separation, self).__init__()
        opt = parse(yaml_path)
        net = Conv_TasNet(**opt['Conv_Tasnet'])
        dicts = torch.load(model, map_location='cpu')
        net.load_state_dict(dicts["model_state_dict"])
        setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
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
        freq = 2048
        snr_change_time = 2000
        batch_size = 1
        temp=0
        test_data = get_saparate_data_test()
        with torch.no_grad():
            for egs, y1,y2 in test_data:
                y1=torch.from_numpy(y1)
                y2=torch.from_numpy(y2)
                if torch.argmax(y1)>=torch.argmax(y2):
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
                # mix_filename=mix_folder+f'_count{m}.png'
                # plt.figure(figsize=(8, 6))
                # plt.plot(egs.cpu().numpy().reshape(8192))
                # plt.title('Mix Signal')
                # if m % 100 == 0:
                #     plt.savefig(mix_filename)
                # plt.close()
                index=0
                for s in spks:
                    s = s[:egs.shape[1]]
                    s = s - torch.mean(s)
                    s = s/torch.max(torch.abs(s))
                    #norm
                    #s = s*norm/torch.max(torch.abs(s))
                    s = s.unsqueeze(0)
                    index += 1
                    # os.makedirs(file_path+'/singnal'+str(index), exist_ok=True)
                    # Saparate_filename = file_path + '/singnal' + str(index) + '/' + f'_spk{index}_count{m}.png'
                    if  index==1:
                         Saparate_filename=file_path+'/singnal1/'+f'_spk{index}_count{m}.png'
                    else:
                         Saparate_filename = file_path + '/singnal2/' + f'_spk{index}_count{m}.png'
                    # 绘制预测结果并保存
                    plt.figure(figsize=(8, 6))
                    plt.plot(s.cpu().numpy().reshape(freq*sample_length))
                    plt.plot(y1.cpu().numpy().reshape(freq*sample_length) if index==1 else y2.cpu().numpy().reshape(freq*sample_length))
                    # plt.plot(y1.reshape(8192) if index == 1 else y2.reshape(8192))
                    plt.title('Saparate_Signal')
                    # plt.plot(s2.cpu().numpy().reshape(8192))
                    # plt.plot(y2.reshape(8192))
                    # plt.legend()
                    # plt.title('Saparate_Signal')
                    if m % 100 == 0:
                        plt.savefig(Saparate_filename)
                    plt.show()
                    plt.close()
                m = m + 1
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '-yaml', type=str, default='./config/Conv_Tasnet/train.yml', help='Path to yaml file.')
    parser.add_argument(
        '-model', type=str, default='./checkpoint/Conv_Tasnet/best.pt', help="Path to model file.")
    parser.add_argument(
        '-gpuid', type=str, default='0', help='Enter GPU id number')
    parser.add_argument(
        '-save_path', type=str, default='./Saparate_task_signal_Conv', help='save result path')
    args=parser.parse_args()
    gpuid=[int(i) for i in args.gpuid.split(',')]
    separation=Separation(args.yaml, args.model, gpuid)
    separation.inference(args.save_path)


if __name__ == "__main__":
    main()