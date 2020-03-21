# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:59:09 2020

@author: JUiscoming
"""
import os
import numpy as np
import scipy.io as sio
import scipy.io.wavfile
import matplotlib.pyplot as plt

import librosa
import librosa.display

class STFT():

    def __init__(self, sampling_rate = 16000, FFT_point = 1024, window = 'hamming', window_length = 0.025, window_shift = 0.010):
        # parameters
        self.sampling_rate = sampling_rate
        self.FFT_point = FFT_point
        self.window_point = int(window_length * sampling_rate)
        self.shift_point = int(window_shift * sampling_rate)
        self.waveform = []
        self.T_bin_num = 0
        self.window = {'hamming': np.hamming(self.window_point),
                       'hanning': np.hanning(self.window_point),
                       'blackman': np.blackman(self.window_point)}[window]
        self.channels = 1
        # spectrogram
        self.spectrogram = []

    def transform_matrix(self, cut_half = True):
        for channel in range(self.channels):    
            time_matrix = np.zeros((self.T_bin_num, self.window_point))
            for time_idx in range(self.T_bin_num):
                time_matrix[time_idx, :] = self.waveform[channel][time_idx * self.shift_point: self.window_point + time_idx * self.shift_point] * self.window
            basis_matrix_real = np.zeros((self.FFT_point // 2 + 1 if cut_half else self.FFT_point, self.window_point))
            basis_matrix_imag = np.zeros((self.FFT_point // 2+ 1 if cut_half else self.FFT_point, self.window_point))
            for basis_idx in range(self.FFT_point // 2 + 1 if cut_half else self.FFT_point):
                basis_matrix_real[basis_idx, :] = self.__basis(basis_idx)[0]
                basis_matrix_imag[basis_idx, :] = self.__basis(basis_idx)[1]
            basis_matrix_real = basis_matrix_real.T
            basis_matrix_imag = basis_matrix_imag.T
            spectrogram_real = np.matmul(time_matrix, basis_matrix_real)**2
            spectrogram_imag = np.matmul(time_matrix, basis_matrix_imag)**2
            spectrogram = (spectrogram_real + spectrogram_imag)**0.5
            # Log Power Spectrogram
            spectrogram = 20 * np.log10(spectrogram + 1e-6)
            print('Channel {}, Time bin: {}, Freq bin: {}'.format(channel, spectrogram.shape[0], spectrogram.shape[1]))
            self.spectrogram.append(np.flip(spectrogram, axis = 1))
      

    def load_wav(self, wav):
        self.sampling_rate, waveform = sio.wavfile.read(wav)
        if len(waveform.shape) > 1:
            self.channels = waveform.shape[1]
            for channel in range(self.channels):
                self.waveform.append(waveform[:, channel])
        else:
            self.waveform.append(waveform)

        self.__zero_padding()

    def __basis(self, freq_idx):
        ang_freq = freq_idx * (2 * np.pi) / self.FFT_point  # freq_idx * (2pi/FFT_point)
        t = np.arange(self.window_point)
        sinusoid_real = np.cos(ang_freq * t)
        sinusoid_imag = np.sin(ang_freq * t)
        exponential_sinusoid = [sinusoid_real, sinusoid_imag]

        return exponential_sinusoid

    def __zero_padding(self):
        remainder = (self.waveform[0].shape[0] - self.window_point) % self.shift_point
        for channel in range(self.channels):
            self.waveform[channel] = np.concatenate((self.waveform[channel], np.array([0] * (self.shift_point - remainder) )), axis = 0)               
        self.T_bin_num = int((len(self.waveform[0]) - self.window_point) / self.shift_point) + 1   # of shift + 1

    def transform(self, cut_half = True):
        self.spectrogram = np.zeros((self.T_bin_num, (self.FFT_point // 2 + 1 if cut_half else self.FFT_point)))

        for time_idx in range(self.T_bin_num):
            waveform_piece = self.waveform[time_idx * self.shift_point: self.window_point + time_idx * self.shift_point]
            # windowed signal
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
        fig_waveform = []
        fig_spectrogram = []
        # plot waveforms
        for channel in range(self.channels):
            fig_waveform.append(fig.add_subplot(self.channels * 2, 1, channel + 1))
            plt.title('Waveform Channel {}'.format(channel))
            fig_waveform[channel].plot(self.waveform[channel], c = 'm')
        # plot spectrograms        
        for channel in range(self.channels):        
            fig_spectrogram.append(fig.add_subplot(self.channels * 2, 1, self.channels + channel + 1))
            plt.title('Spectrogram Channel {}'.format(channel))
            plt.xlabel('Time bin')
            plt.ylabel('Freq bin')
            plt.yticks([0, self.FFT_point//4, self.FFT_point//2], [self.FFT_point//2, self.FFT_point//4, 0])
            fig_spectrogram[channel].imshow(self.spectrogram[channel].T)

        plt.tight_layout()
        plt.show()
