'''
revise frequency, just change freq = XXX
generate data with different chirp mass







'''
import argparse
import gc
import time
import os

import matplotlib
# from readconfig import readconfig
import random
from pycbc.waveform import get_td_waveform
import numpy as np
from pycbc.detector import Detector
import pycbc.psd

import pycbc.noise
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.types import TimeSeries

import matplotlib
import matplotlib.pyplot as plt
import pickle
matplotlib.use('TkAgg')

from pycbc.filter import matched_filter

from pycbc.conversions import  mass1_from_mchirp_q, mass2_from_mchirp_q

freq = 4096


delta_f = 1.0 / 64
flen = int(freq/2.0/ delta_f) + 1
low_frequency_cutoff = 10.0
psd = pycbc.psd.EinsteinTelescopeP1600143(flen, delta_f, low_frequency_cutoff)
# Generate 32 seconds of noise at 4096 Hz
delta_t = 1.0 / freq
tsamples = int(64 / delta_t)

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
    if len(np_signal)>=len(np_noise)//2:
        np_signal = np_signal[-len(np_noise)//2+freq:-1]
    signal_pad = pad_to_length(np_signal,len(np_noise))
    # pp.plot(signal_pad)
    # pp.show()
    signal_plus_noise = signal_pad+np_noise
    template_use = shift_max(signal_pad)
    snr = np.abs(matched_filter(template=TimeSeries(template_use,delta_t=1/freq), data=TimeSeries(signal_plus_noise,delta_t=1/freq),psd=psd,low_frequency_cutoff=20,high_frequency_cutoff=1024))
    # pp.plot(snr[len(signal):-1-len(signal)])
    # pp.show()
    # print('snr='+str(snr))
    return max(snr[len(np_signal):-1-len(np_signal)])



det_E1 = Detector('E1')


def generate_mass_list(chirp_mass_low, chirp_mass_high):
    signal_chirp_mass=range(chirp_mass_low,chirp_mass_high,1)
    return signal_chirp_mass





def cut_signal(signal, peak_loc, time_duration, freq):
    peak_num = int(peak_loc * time_duration * freq)
    signal_peak = np.argmax(signal)
    all_num = time_duration * freq
    print(all_num)
    signal_after_peak = np.size(signal) - signal_peak
    after_peak = all_num - peak_num
    return_data = np.zeros(all_num)
    if (peak_num < signal_peak):
        return_data[:peak_num] = signal[signal_peak - peak_num:signal_peak]
    else:
        return_data[peak_num - signal_peak:peak_num] = signal[0:signal_peak]
    if (signal_after_peak < after_peak):
        # print(signal_after_peak)
        # print(after_peak)
        return_data[peak_num:peak_num + signal_after_peak] = signal[
                                                             signal_peak:signal_peak + signal_after_peak]
    else:
        return_data[peak_num:] = signal[signal_peak:signal_peak + after_peak]

    return return_data




def generate_signal_list(chirp_mass_low, chirp_mass_high, snr_value):
    signal_chirp_mass_list = generate_mass_list(chirp_mass_low, chirp_mass_high)
    param_list = []
    signal_list = []
    for chirp_mass in signal_chirp_mass_list:
        mass1 = mass1_from_mchirp_q(chirp_mass,2)
        mass2 = mass2_from_mchirp_q(chirp_mass,2)
        spin1z = random.uniform(0, 0.99)
        spin2z = random.uniform(0, 0.99)
        declination = np.arcsin(random.uniform(0, 2) - 1.0)
        right_ascension = random.uniform(0, np.pi * 2)
        polarization = random.uniform(0, np.pi * 2)
        coa_phase = random.uniform(0, np.pi * 2)
        psd = pycbc.psd.EinsteinTelescopeP1600143(flen, delta_f, low_frequency_cutoff)
        noise = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=random.randint(1, 1000000000))
        param = {}
        param['spin1z'] = spin1z
        param['spin2z'] = spin2z
        param['declination'] = declination
        param['right_ascension'] = right_ascension
        param['polarization'] = polarization
        param['coa_phase'] = coa_phase
        hp, hc = get_td_waveform(approximant='SEOBNRv4',
                                 mass1=mass1,
                                 mass2=mass2,
                                 spin1z=spin1z,
                                 spin2z=spin2z,
                                 delta_t=1.0 / freq,
                                 f_lower=10,
                                 coa_phase=coa_phase,
                                 inclination=0.5,
                                 distance=4000)
        signal = det_E1.project_wave(hp, hc, right_ascension, declination, polarization)

        signal = cut_signal(signal, 0.8, 4, 4096)
        snr = matchSNR(signal, noise, psd)
        # print('snr={}'.format(snr))
        signal = signal/snr*snr_value


        signal_list.append(signal)
        param_list.append(param)

        # plt.plot(signal)
        # plt.show()
    noise1 = noise[4096:4096 + 4 * 4096]
    return signal_chirp_mass_list, signal_list, noise1, param_list


# signal_list[i]'s chirp mass is chirp_mass_list[i]
# only one noise is shown
# the snr of signal is controled by snr_value in parameter of function generate_signal_list
chirp_mass_list, signal_list, noise, param = generate_signal_list(10, 81, 10)
print(len(chirp_mass_list))
print(len(signal_list))
with open('new_test_chirpmass_data.pkl', 'wb') as f:
    pickle.dump((chirp_mass_list, signal_list, param), f)
