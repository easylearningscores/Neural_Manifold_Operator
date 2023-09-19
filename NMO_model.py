import torch
from torch import nn
from modules import ConvSC, Inception
from fouriermodules import *
from evolution import Spatio_temporal_evolution
import math



def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class Downsampling(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Downsampling,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )
    
    def forward(self,x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1


class Upsampling(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Upsampling,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y

class Temporal_evo(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3, 5, 7, 11], groups=8):
        super(Temporal_evo, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T - 1):
            enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T - 1):
            dec_layers.append(
                Inception(2 * channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers.append(Inception(2 * channel_hid, channel_hid // 2, channel_in, incep_ker=incep_ker, groups=groups))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(channel_hid)

        self.enc = nn.Sequential(*enc_layers)
        dpr = [x.item() for x in torch.linspace(0, 0, 12)]
        self.blocks = nn.ModuleList([FourierNetBlock(
            dim=channel_hid,
            mlp_ratio=4,
            drop=0.,
            drop_path=dpr[i],
            act_layer=nn.GELU,
            norm_layer=norm_layer,
            h=32,
            w=32)
            for i in range(12)
        ])
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        bias = x
        x = x.reshape(B, T * C, H, W)

        # downsampling
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)
        
        # Spectral Domain
        B, D, H, W = z.shape
        N = H * W
        z = z.permute(0, 2, 3, 1)
        z = z.view(B, N, D)
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z).permute(0, 2, 1)
        z = z.reshape(B, D, H, W)

        # upsampling
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y + bias

import torch.nn as nn

class time_step_projection(nn.Module):
    def __init__(self, C, input_time_step, output_time_step):
        super(time_step_projection, self).__init__()
        self.linerprojection = nn.Conv2d(C * input_time_step,
                                         C * output_time_step,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         bias=False)
        self.group = nn.GroupNorm(2, C * output_time_step)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.linerprojection(x)
        x = self.group(x)
        x = self.act(x)
        return x

    
class neural_manifold_operator(nn.Module):
    def __init__(self, shape_in, id_dim=4, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8, input_time_step=10, output_time_step=90):
        super(neural_manifold_operator, self).__init__()
        T, C, H, W = shape_in
        self.H1 = int(H / 2 ** (N_S / 2)) + 1 if H % 3 == 0 else int(H / 2 ** (N_S / 2))
        self.W1 = int(W / 2 ** (N_S / 2))
        self.input_time_step =input_time_step
        self.output_time_step = output_time_step
        self.downconv = Downsampling(C, id_dim, N_S)
        self.hid = Temporal_evo(T*id_dim, hid_T, N_T, incep_ker, groups) #
        self.temporal_evolution = Spatio_temporal_evolution(T*id_dim, hid_T, N_T,
                                                            input_resolution=[self.H1, self.W1],
                                                            model_type='gsta',
                                                            mlp_ratio=4.,
                                                            drop=0.0,
                                                            drop_path=0.1)

        self.upconv = Upsampling(id_dim, C, N_S)
        self.timeprojection = time_step_projection(C, input_time_step=input_time_step, output_time_step=output_time_step)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)
        

        embed, skip = self.downconv(x)
        _, C_, H_, W_ = embed.shape

        #print("embed shape", embed.shape)

        z = embed.view(B, T, C_, H_, W_)
        bias = z
        bias_hid = self.hid(bias)
        bias_hid = bias_hid.reshape(B*T, C_, H_, W_)
        hid = self.temporal_evolution(z)
        hid = hid.reshape(B*T, C_, H_, W_)
        hid = hid+bias_hid

        #reconstruction
        x_rec = embed
        x_rec = self.upconv(x_rec, skip)
        x_rec = x_rec.reshape(B, T, C, H, W)

        Y = self.upconv(hid, skip)
        Y = Y.reshape(B, T, C, H, W).view(B, T*C, H,W)
        Y = self.timeprojection(Y)
        Y = Y.reshape(B, 90, C, H, W)
        return Y, x_rec

if __name__ == '__main__':
    inputs = torch.randn(16, 10, 1, 128, 128)
    model = neural_manifold_operator(shape_in=(10, 1, 128, 128))
    output, x_rec = model(inputs)

    print(output.shape, x_rec.shape)