'''
revise frequency, just change freq = XXX








'''
import argparse
import gc
import time
import os

import matplotlib
from readconfig import readconfig
import random
from pycbc.waveform import get_td_waveform
import numpy as np
from pycbc.detector import Detector
import pycbc.psd

import pycbc.noise
import matplotlib.pyplot as pp
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.types import TimeSeries
import matplotlib
import pickle
matplotlib.use('TkAgg')

from pycbc.filter import matched_filter

freq = 2048
output_file_name = 'data1.pkl'

def pad_to_length(arr,length):
    left = length//2-np.argmax(arr)
    mid = arr.size
    right = length-left-mid
    return np.pad(arr,(left,right),'constant',constant_values=0)

def shift_max(vector):
    shift_amount = len(vector)-np.argmax(vector)-1
    shift_amount %= len(vector)
    return np.roll(vector,shift=shift_amount)


def matchSNR(signal, noise, psd): # signal's length is smaller than noise
    np_signal = np.array(signal)
    np_noise = np.array(noise)
    signal_pad = pad_to_length(np_signal,len(np_noise))
    signal_plus_noise = signal_pad+np_noise
    template_use = shift_max(signal_pad)
    snr = np.abs(matched_filter(template=TimeSeries(template_use,delta_t=1/freq), data=TimeSeries(signal_plus_noise,delta_t=1/freq),psd=psd,low_frequency_cutoff=20,high_frequency_cutoff=1024))
    # pp.plot(snr[len(signal):-1-len(signal)])
    # pp.show()
    # print('snr='+str(snr))
    return max(snr[len(signal):-1-len(signal)])



det_E1 = Detector('E1')
det_E2 = Detector('E2')
det_E3 = Detector('E3')

def generate_antena_response(mass1_range, mass2_range, spin1z_range, spin2z_range,noiseE1,noiseE2,noiseE3,psd):

    random.seed(time.time())
    mass1_list = []
    mass2_list = []
    spin1z_list = []
    spin2z_list = []
    right_ascension_list = []
    declination_list = []
    signal_E1_list = []
    signal_E2_list = []
    signal_E3_list = []
    snr_E1_list = []
    snr_E2_list = []
    snr_E3_list = []

    for ii in range(10):
        mass1 = random.uniform(mass1_range[0],mass1_range[1])
        mass2 = random.uniform(mass2_range[0],mass2_range[1])
        spin1z = random.uniform(spin1z_range[0],spin1z_range[1])
        spin2z = random.uniform(spin2z_range[0],spin2z_range[1])
        declination = np.arcsin(random.uniform(0, 2)-1.0)
        right_ascension = random.uniform(0, np.pi*2)
        polarization = random.uniform(0,np.pi*2)
        coa_phase = random.uniform(0,np.pi*2)

        hp, hc = get_td_waveform(approximant='SEOBNRv4',
                                 mass1=mass1,
                                 mass2=mass2,
                                 spin1z=spin1z,
                                 spin2z=spin2z,
                                 delta_t=1.0 / freq,
                                 f_lower=10,
                                 coa_phase=coa_phase,
                                 distance=4000)
        signal_E1 = det_E1.project_wave(hp, hc, right_ascension, declination, polarization)
        signal_E2 = det_E2.project_wave(hp, hc, right_ascension, declination, polarization)
        signal_E3 = det_E3.project_wave(hp, hc, right_ascension, declination, polarization)
        print('noise='+str(noiseE1))
        print('mass1 = '+str(mass1))
        print('mass2 = '+str(mass2))
        print('psd='+str(psd))
        mass1_list.append(mass1)
        mass2_list.append(mass2)
        spin1z_list.append(spin1z)
        spin2z_list.append(spin2z)
        right_ascension_list.append(right_ascension)
        declination_list.append(declination)
        signal_E1_list.append(np.array(signal_E1))
        signal_E2_list.append(np.array(signal_E2))
        signal_E3_list.append(np.array(signal_E3))
        snr_E1_list.append(matchSNR(signal_E1,noiseE1,psd))
        snr_E2_list.append(matchSNR(signal_E2, noiseE2, psd))
        snr_E3_list.append(matchSNR(signal_E3, noiseE3, psd))
    return mass1_list, mass2_list, spin1z_list,spin2z_list, right_ascension_list, declination_list, signal_E1_list, signal_E2_list, signal_E3_list, snr_E1_list, snr_E2_list, snr_E3_list

delta_f = 1.0 / 64
flen = int(freq/2.0/ delta_f) + 1
low_frequency_cutoff = 10.0
psd = pycbc.psd.EinsteinTelescopeP1600143(flen, delta_f, low_frequency_cutoff)
# Generate 32 seconds of noise at 4096 Hz
delta_t = 1.0 / freq
tsamples = int(64 / delta_t)

def generate_noise():

    noiseE1 = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=random.randint(1,1000000000))
    noiseE2 = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=random.randint(1,1000000000))
    noiseE3 = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=random.randint(1, 1000000000))
    return noiseE1,noiseE2,noiseE3

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
        print('noiseE1, noiseE2, noiseE3 are 32 s length noise of E1, E2 and E3')
        print('signal_E1_list, signal_E2_list and signal_E3_list are signals, and each have 10 samples')
        print('mass1_list, mass2_list are masses, and each have 10 masses')
        print('right_ascension_list and declination_list are the directions of the source of the signal')





if __name__ == '__main__':
    # matplotlib.use('TkAgg')
    # parser = argparse.ArgumentParser(description='The name of the config file XXX.yml')
    # parser.add_argument('--configfile', type=str, default='config.yml', help='--configfile default is config.yml')
    # args = parser.parse_args()
    # start_time = time.time() # start of the time
    #
    # # the full path of the config file
    # yaml_config_file = args.configfile
    # yaml_config_path = os.path.join('.','config',yaml_config_file)
    #
    # # read the config file return a dictionary. The config file tell us:
    # # the range of the mass, spin, snr, and the input_dir(strain without GW), output_dir
    # config = readconfig(yaml_config_path)
    # output_dir = os.path.join('.', config['output_dir'])

    data = []

    for index in range(1000):
        noiseE1, noiseE2, noiseE3 = generate_noise()
        mass1_list, mass2_list, spin1z_list, spin2z_list, right_ascension_list, declination_list, signal_E1_list, signal_E2_list, signal_E3_list, snr_E1_list, snr_E2_list, snr_E3_list= (
            generate_antena_response((20,80), (20,80), (0,0.998), (0,0.998), noiseE1, noiseE2, noiseE3, psd))
        data.append(data_sample(np.array(noiseE1),np.array(noiseE2),np.array(noiseE3),mass1_list,mass2_list,spin1z_list,spin2z_list,right_ascension_list,
                                declination_list,signal_E1_list,signal_E2_list,signal_E3_list,snr_E1_list,snr_E2_list,snr_E3_list))

    with open('output/'+output_file_name, 'wb') as f:
        pickle.dump(data, f)