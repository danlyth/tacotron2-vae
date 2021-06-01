from torch import nn
import torch
import numpy as np


class Tacotron2Loss_VAE(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss_VAE, self).__init__()
        self.anneal_function = hparams.anneal_function
        self.lag = hparams.anneal_lag
        self.k = hparams.anneal_k
        self.x0 = hparams.anneal_x0
        self.upper = hparams.anneal_upper
    
    def kl_anneal_function(self, anneal_function, lag, step, k, x0, upper):
        if anneal_function == 'logistic':
            return float(upper/(upper+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            if step > lag:
                return min(upper, step/x0)
            else:
                return 0
        elif anneal_function == 'constant': # simple beta-VAE
            return 0.001


    def forward(self, model_output, targets, step): # NOTE model ouput==y_pred in train.py
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _, mu, logvar, _, _ = model_output # NOTE mu and logvar are both nan, mel_out, mel_out_postnet are all 0 and gate_out is all 1000
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        # print(f'Log var: {logvar}, \nMu: {mu}, \nMu power: {mu.pow(2)}, \nLog var exp: {logvar.exp()}')
        # print(f'mel_out: {mel_out}, \nmel_out_posnet: {mel_out_postnet}, \ngate_out: {gate_out}')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_weight = self.kl_anneal_function(self.anneal_function, self.lag, step, self.k, self.x0, self.upper)
        
        recon_loss = mel_loss + gate_loss
        total_loss = recon_loss + kl_weight*kl_loss
        # print(f'Total loss:{total_loss}, Recon loss: {recon_loss}, KL loss: {kl_loss}, KL weight: {kl_weight}')
        return total_loss, recon_loss, kl_loss, kl_weight
