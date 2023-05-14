import pandas as pd
import numpy as np
from scipy import signal
from scipy.fftpack import dct
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pywt
import os


# Set the path to the Excel file
excel_file_path = 'D:\HCI\Data.xlsx'

# Read the Excel file into a pandas DataFrame object
excel_data = pd.read_excel(excel_file_path, sheet_name=None)

def Batlhause_filter(Data):

    # Apply Batlhause filtering to the ECG signals
    sampling_rate = 1000 # Hz
    notch_freq = 60 # Hz
    notch_width = 2 # Hz
    b, a = signal.iirnotch(notch_freq / (sampling_rate / 2), notch_width)
    ecg_filtered = signal.filtfilt(b, a, Data, axis=0)

    return ecg_filtered

def Z_score_Normalization(Data):

    # Apply z-score normalization to the ECG signals
    mean = np.mean(Data, axis=0)
    std = np.std(Data, axis=0)
    ecg_normalized = (Data - mean) / std

    return ecg_normalized

def Wavelet_transform(Data):

    # Apply wavelet transform to the ECG signals
    wavelet_name = 'db4'
    level = 5
    coeffs = pywt.wavedec(Data, wavelet_name, level=level)
    cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
    
    # Extract features from the wavelet coefficients
    peak_amplitude = np.max(np.abs(coeffs), axis=1)
    peak_frequency = np.argmax(np.abs(coeffs), axis=1) / 2**level
    waveform_width = np.sum(np.abs(coeffs), axis=1)
    
    return peak_amplitude, peak_frequency, waveform_width

def DCT_Extraction(Data):

    # Apply DCT to the signal
    dct_coeffs = dct(Data,type=2, norm='ortho')
        
    # Normalize the coefficients
    norm_coeffs = (dct_coeffs - np.mean(dct_coeffs)) / np.std(dct_coeffs)

    return norm_coeffs

def ACF_Extraction(Data):

    # Apply ACF to the signal
    acf = sm.tsa.acf(Data, nlags=len(Data)-1)
    acf = acf[:1000]

    return acf

# Create an ExcelWriter object to write the data to a single file
writer = pd.ExcelWriter('PreProcessed_Data_ACF_DCT.xlsx', engine='xlsxwriter')

for sheet_name, sheet_data in excel_data.items():
    
    df = pd.DataFrame()
    
    for signal_col in sheet_data.columns:

        signal_data = sheet_data[signal_col].to_numpy()
        DCT = DCT_Extraction(ACF_Extraction(Z_score_Normalization(Batlhause_filter(signal_data))))
        #amplitude, frequency, Width = Wavelet_transform(Z_score_Normalization(Batlhause_filter(signal_data)))

        # Plot the extracted features
        """plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.plot(DCT)
        plt.title(f'DCT Coefficients for Signal {signal_col} in Sheet {sheet_name}')
        plt.xlabel('Coefficient Index')
        plt.ylabel('Normalized Coefficient Value')
        plt.subplot(1, 2, 2)
        plt.plot(signal_data)
        plt.title(f'Signal {signal_col} from {sheet_name}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.show()
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(amplitude, 'o-')
        plt.title('Peak Amplitude')
        plt.xlabel('Wavelet Coefficient')
        plt.ylabel('Magnitude')
        plt.subplot(2, 2, 2)
        plt.plot(frequency, 'o-')
        plt.title('Peak Frequency')
        plt.xlabel('Wavelet Coefficient')
        plt.ylabel('Frequency (Hz)')
        plt.subplot(2, 2, 3)
        plt.plot(Width, 'o-')
        plt.title('Waveform Width')
        plt.xlabel('Wavelet Coefficient')
        plt.ylabel('Magnitude')
        plt.subplot(2, 2, 4)
        plt.plot(signal_data, 'o-')
        plt.title(f'Signal {signal_col} from {sheet_name}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')"""
        #plt.show()
        # Add the ECG data to the DataFrame
        df[f'ECG_{signal_col}'] = DCT[:]

    # Add the DataFrame as a sheet in the Excel file
    df.to_excel(writer, sheet_name=sheet_name, index=False)

# Save the Excel file
writer.close()