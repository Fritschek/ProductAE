import math
import logging

from tqdm.notebook import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import utils




class TrainerConfig:
    # optimization parameters, gets overriden by config init
    max_epochs = 10
    learning_rate = 1e-4
    rate = 0.5

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, encoder, decoder, channel, config):
        self.channel = channel
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train(self):
        channel, config, encoder, decoder = self.channel, self.config, self.encoder, self.decoder
        #optimizer = torch.optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
        lr = config.learning_rate
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
        #scheduler_enc = StepLR(encoder_optimizer, step_size=30, gamma=0.5) # Every 30 Epochs, reduce lr by factor 0.75
        #scheduler_dec = StepLR(decoder_optimizer, step_size=30, gamma=0.5)
        self.K = config.K
        input_dim = int(self.K[0]*self.K[1])
        loss_CE = nn.CrossEntropyLoss()
        loss_epoch = []
        best_loss = float('inf')
        def run_epoch(batch_size):         
            
            loss_tot, losses, batch_BER, batch_BLER,   = [], [], [], []     

            for i in range(100):  # Train encoder 100 times
                encoder_optimizer.zero_grad()
                    
                m = torch.randint(0,2, size=(batch_size, input_dim))
                m = m.float().to(self.device)
                
                x = encoder(m)
                x_out = channel(x, config.noise_std)
                output = decoder(x_out)
                
                loss = F.binary_cross_entropy(output, m)
                loss = loss.mean()
                loss.backward()
                losses.append(loss.item())
                loss_tot.append(loss.item())
                encoder_optimizer.step()

            for i in (pbar := tqdm(range(500))):  # Train decoder 500 times
                decoder_optimizer.zero_grad()
                    
                m = torch.randint(0,2, size=(batch_size, input_dim))
                m = m.float().to(self.device)  
                x = encoder(m)
                
                # This part draws a noise std matrix according to uniformly dist EbN0/SNR range, which gets fed to the channel func
                low_snrdb = config.training_snr - 2.5
                high_snrdb = config.training_snr + 1
                i,j = x.size()
                snrdb_matrix = np.random.uniform(low_snrdb, high_snrdb, size=(i, j))
                noise_std_matrix = utils.SNR_to_noise(snrdb_matrix)
                noise_std_matrix_torch = torch.from_numpy(noise_std_matrix).float().to(self.device)
                
                x_out = channel(x, noise_std_matrix_torch)
                output = decoder(x_out)
                
                loss = F.binary_cross_entropy(output, m)
                loss = loss.mean()
                loss.backward()
                losses.append(loss.item())
                loss_tot.append(loss.item())
                decoder_optimizer.step()
                    
                ber, ser = utils.error_binary(torch.round(output), m)
                batch_BER.append(ber)
                batch_BLER.append(ser)
                pbar.set_description(f"epoch {epoch+1} : loss {np.mean(losses):.3e} BER {np.mean(batch_BER[-100:]):.3e} BLER {np.mean(batch_BLER[-100:]):.3e}")

            return loss_tot
    
        for epoch in range(config.max_epochs):
            
            loss_tot = run_epoch(config.batch_size+5*epoch) # pushing batch size higher with epochs
            loss_tot_avg = np.mean(loss_tot)
            loss_epoch.append(loss_tot_avg)
            #scheduler_enc.step()
            #scheduler_dec.step()
            
            if loss_tot_avg < best_loss:
                best_loss = loss_tot_avg
                torch.save({ 'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict()}, config.path)
                
        return loss_epoch

    
    def test(self, snr_range, iterations, batch_size, K):
        channel, config, encoder, decoder = self.channel, self.config, self.encoder, self.decoder
        ber, bler = [], []
        its = int(iterations)
                
        def run_epoch():         
            batch_BER, batch_BLER = [], []
                        
            for it in (pbar := tqdm(range(its))):
                
                input_dim = int(K[0]*K[1])
                m = torch.randint(0,2, size=(batch_size, input_dim))
                m = m.float().to(self.device)
                noise_std = utils.SNR_to_noise(snr_range[epoch])
                with torch.no_grad():  # Disable gradient calculation
                    x = encoder(m)
                    x_out = channel(x, noise_std)
                    output = decoder(x_out)
                
                ber_, bler_ = utils.error_binary(torch.round(output), m)
                batch_BER.append(ber_)
                batch_BLER.append(bler_)
                pbar.set_description(f"SNR: {snr_range[epoch]} BER {np.mean(batch_BER[-100:]):.3e} BLER {np.mean(batch_BLER[-100:]):.3e}")
                    
       
            return np.mean(batch_BER), np.mean(batch_BLER)
            
        for epoch in range(len(snr_range)):
            
            ber_i, bler_i = run_epoch()
            ber.append(ber_i)
            bler.append(bler_i)
        
        return ber, bler