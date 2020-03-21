# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:59:09 2020

@author: JUiscoming
"""

import numpy as np
import scipy.io as sio
import scipy.io.wavfile
import matplotlib.pyplot as plt

class STFT():

    def __init__(self, sampling_rate = 16000, FFT_point = 256, window = 'hamming', window_length = 0.025, window_shift = 0.010):
        #parameters
        self.sampling_rate = sampling_rate
        self.FFT_point = FFT_point
        self.window_point = int(window_length * sampling_rate)
        self.shift_point = int(window_shift * sampling_rate)
        self.waveform = None
        self.T_bin_num = 0
        #windows
        self.window = {'hamming': np.hamming(self.window_point),
                       'hanning': np.hanning(self.window_point),
                       'blackman': np.blackman(self.window_point)}[window]
        #spectrogram
        self.spectrogram = None

    def load_wav(self, wav):
        self.sampling_rate, self.waveform = sio.wavfile.read(wav)
        if len(self.waveform.shape) > 1:
            self.waveform = np.sum(self.waveform, axis = 1) / self.waveform.shape[1]
        self.waveform = self.waveform.astype(np.float32)

        self.__zero_padding()

    def __basis(self, freq_idx):
        ang_freq = freq_idx * (2 * np.pi) / self.FFT_point  #freq_idx * (2pi/FFT_point)
        t = np.arange(self.window_point)
        sinusoid_real = np.cos(ang_freq * t)
        sinusoid_imag = np.sin(ang_freq * t)
        exponential_sinusoid = [sinusoid_real, sinusoid_imag]

        return exponential_sinusoid

    def __zero_padding(self):
        remainder = (len(self.waveform) - self.window_point) % self.shift_point
        if remainder:
            self.waveform = np.concatenate((self.waveform, np.array([0] * (self.shift_point - remainder) )), axis = 0)
        self.T_bin_num = int((len(self.waveform) - self.window_point) / self.shift_point) + 1   # of shift + 1

    def transform(self, cut_half = True):
        self.spectrogram = np.zeros((self.T_bin_num, (self.FFT_point // 2 + 1 if cut_half else self.FFT_point)))

        for time_idx in range(self.T_bin_num):
            waveform_piece = self.waveform[time_idx * self.shift_point: self.window_point + time_idx * self.shift_point]
            #windowed signal
            waveform_piece = waveform_piece * self.window

            for freq_idx in range(self.FFT_point // 2 + 1 if cut_half else self.FFT_point):
                b = self.__basis(freq_idx)
                self.spectrogram[time_idx, freq_idx] = (np.sum(waveform_piece * b[0])**2 + np.sum(waveform_piece * b[1])**2)**0.5

        # log scale
        self.spectrogram = 20 * np.log10(self.spectrogram + 1e-6)
        print('Time bin: {}, Freq bin: {}'.format(self.spectrogram.shape[0], self.spectrogram.shape[1]))
        self.spectrogram = np.flip(self.spectrogram, axis = 1)

    def plot(self):
        fig = plt.figure()
        fig1 = fig.add_subplot(211)
        plt.title('Waveform')
        fig1.plot(self.waveform, c = 'm')

        fig2 = fig.add_subplot(212)
        plt.title('Spectrogram')
        plt.xlabel('Time bin')
        plt.ylabel('Freq bin')
        plt.yticks([0, self.FFT_point//4, self.FFT_point//2], [self.FFT_point//2, self.FFT_point//4, 0])
        fig2.imshow(self.spectrogram.T)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    stft = STFT()
    stft.load_wav('test.wav')
    stft.transform()
    stft.plot()
