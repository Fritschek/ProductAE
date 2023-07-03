import sys
assert sys.version_info >= (3, 5)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
from scipy import special, integrate
from scipy.spatial import distance
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
np.random.seed(42)



def EbNo_to_noise(ebnodb, rate):
    '''Transform EbNo[dB]/snr to noise power'''
    ebno = 10**(ebnodb/10)
    noise_std = 1/np.sqrt(rate*ebno) #without factor two, as in the original paper
    return noise_std

def SNR_to_noise(snrdb):
    '''Transform EbNo[dB]/snr to noise power'''
    snr = 10**(snrdb/10)
    noise_std = 1/np.sqrt(snr) # without the factor two as in the original paper
    return noise_std


def error_binary(x_hat, x):
    prediction_errors = torch.ne(x_hat, x)
    Bit_error_vec = torch.sum(prediction_errors, dim=1)

    # Compute BER
    BER = torch.mean(prediction_errors.float())

    # Compute SER (any symbol with one or more bit errors is considered incorrect)
    SER = torch.mean((Bit_error_vec > 0).float())

    return BER.detach().cpu().item(), SER.detach().cpu().item()

def plot_train_loss(train_losses):
    # Create a DataFrame

    df = pd.DataFrame({'train_losses': train_losses})

    span = 50

    # Calculate the Exponential Moving Average (EMA)
    df['train_losses_ema'] = df['train_losses'].ewm(span=span).mean()

    plt.figure()

    # Plot training losses
    plt.plot(df['train_losses'], label='Training Loss')
    plt.plot(df['train_losses_ema'], label='Training Loss EMA')
    plt.title('Training Losses')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # Save and display the figure
    plt.tight_layout()
    plt.savefig('train_losses.png', dpi=300)
    plt.show()
