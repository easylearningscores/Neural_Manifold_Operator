import torch
from torch import nn
from fouriermodules import *
from evolution import Spatio_temporal_evolution
import math
from torch import nn


class Inception(nn.Module):
    def __init__(self, input_dim, state_dim, output_dim, incep_ker=[3, 5, 7, 11], groups=8, act_norm=False):
        super(Inception, self).__init__()
        self.act_norm = act_norm
        if state_dim % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(input_dim, state_dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.GroupNorm(groups, state_dim)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        
        layers = []
        for ker in incep_ker:
            layers.append(nn.Conv2d(state_dim, output_dim, kernel_size=ker, stride=1, padding=ker//2, groups=groups))
        self.inception_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        if self.act_norm:
            x = self.activate(self.norm(x))
        y = 0
        for layer in self.inception_layers:
            y += layer(x)
        return y

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]


class Encoder(nn.Module):
    def __init__(self, input_dim, state_dim, num_layers):
        super(Encoder, self).__init__()
        strides = stride_generator(num_layers)

        self.conv0 = nn.Conv2d(input_dim, state_dim, kernel_size=3, stride=strides[0], padding=1)
        self.norm0 = nn.GroupNorm(2, state_dim)
        self.act0 = nn.LeakyReLU(0.2, inplace=True)
        

        self.convs = nn.ModuleList([nn.Conv2d(state_dim, state_dim, kernel_size=3, stride=s, padding=1) for s in strides[1:]])
        self.norms = nn.ModuleList([nn.GroupNorm(2, state_dim) for s in strides[1:]])
        self.acts = nn.ModuleList([nn.LeakyReLU(0.2, inplace=True) for s in strides[1:]])

    def forward(self, x):
        x = self.act0(self.norm0(self.conv0(x)))
        for conv, norm, act in zip(self.convs, self.norms, self.acts):
            x = act(norm(conv(x)))
            
        return x

class Decoder(nn.Module):
    def __init__(self, state_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        strides = stride_generator(num_layers, reverse=True)
        
        self.deconvs = nn.ModuleList([nn.ConvTranspose2d(state_dim, state_dim, kernel_size=3, stride=s, padding=1, output_padding=s//2) for s in strides[:-1]])
        self.denorms = nn.ModuleList([nn.GroupNorm(2, state_dim) for s in strides[:-1]])
        self.deacts = nn.ModuleList([nn.LeakyReLU(0.2, inplace=True) for s in strides[:-1]])

        self.deconv_final = nn.ConvTranspose2d(2*state_dim, state_dim, kernel_size=3, stride=strides[-1], padding=1, output_padding=strides[-1]//2)
        self.denorm_final = nn.GroupNorm(2, state_dim)
        self.deact_final = nn.LeakyReLU(0.2, inplace=True)

        self.readout = nn.Conv2d(state_dim, output_dim, 1)

    def forward(self, state):
        for deconv, denorm, deact in zip(self.deconvs, self.denorms, self.deacts):
            state = deact(denorm(deconv(state)))

        state = self.deact_final(self.denorm_final(self.deconv_final(state)))
        Y = self.readout(state)
        return Y

 
class Temporal_evo(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, h, w, incep_ker=[3, 5, 7, 11], groups=8):
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
        self.h = h
        self.w = w
        self.blocks = nn.ModuleList([FourierNetBlock(
            dim=channel_hid,
            mlp_ratio=4,
            drop=0.,
            drop_path=dpr[i],
            act_layer=nn.GELU,
            norm_layer=norm_layer,
            h = self.h,
            w = self.w)
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
    
class neural_manifold_operator(nn.Module):
    def __init__(self, shape_in, hid_S = 64, id_dim=4, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8, in_time_seq_length=13, out_time_seq_length=12):
        super(neural_manifold_operator, self).__init__()
        T, C, H, W = shape_in
        self.H1 = int(H / 2 ** (N_S / 2)) + 1 if H % 3 == 0 else int(H / 2 ** (N_S / 2))
        self.W1 = int(W / 2 ** (N_S / 2))
        self.in_time_seq_length = in_time_seq_length
        self.out_time_seq_length = out_time_seq_length
        
        self.Enconv = Encoder(C, hid_S, N_S)
        self.high_to_id = nn.Conv2d(hid_S, id_dim, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.hid = Temporal_evo(T*id_dim, hid_T, N_T, self.H1, self.W1, incep_ker, groups) #
        self.temporal_evolution = Spatio_temporal_evolution(T*id_dim, hid_T, N_T,
                                                            input_resolution=[self.H1, self.W1],
                                                            model_type='gsta',
                                                            mlp_ratio=4.,
                                                            drop=0.0,
                                                            drop_path=0.1)
        
        self.id_to_high = nn.Conv2d(id_dim, hid_S, kernel_size=1, stride=1, padding=0, bias=False)
        self.Deconv = Decoder(hid_S, C, N_S)

    def forward_(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)
        embed, skip = self.Enconv(x)


        
        id_dim = self.high_to_id(embed)
        _, id, H_, W_ = id_dim.shape


        z = id_dim.view(B, T, id, H_, W_)
        bias = z
        bias_hid = self.hid(bias)
        bias_hid = bias_hid.reshape(B*T, id, H_, W_)
    

        hid = self.temporal_evolution(z)
        hid = hid.reshape(B*T, id, H_, W_)
        hid = bias_hid + hid
        #print("hid shape", hid.shape)

       
        x_rec = embed
        x_rec = self.Deconv(x_rec, skip)
        x_rec = x_rec.reshape(B, T, C, H, W)

        hid  = self.id_to_high(hid)
        Y = self.Deconv(hid, skip)
        Y = Y.reshape(B, T, C, H, W).view(B, T*C, H,W)

      
        Y = Y.reshape(B, T, C, H, W)
        return Y, x_rec

    def forward(self, xx):
        yy, x_rec = self._forward(xx)
        in_time_seq_length, out_time_seq_length = self.in_time_seq_length, self.out_time_seq_length
        if out_time_seq_length == in_time_seq_length:
            y_pred = yy
        if out_time_seq_length < in_time_seq_length:
            y_pred = yy[:, :out_time_seq_length]
        elif out_time_seq_length > in_time_seq_length:
            y_pred = [yy]
            d = out_time_seq_length // in_time_seq_length
            m = out_time_seq_length % in_time_seq_length
            
            for _ in range(1, d):
                cur_seq, x_rec = self._forward(y_pred[-1])
                y_pred.append(cur_seq)
            
            if m != 0:
                cur_seq, x_rec = self._forward(y_pred[-1])
                y_pred.append(cur_seq[:, :m])
            
            y_pred = torch.cat(y_pred, dim=1)
        
        return y_pred, x_rec

if __name__ == '__main__':
    inputs = torch.randn(16, 10, 1, 64, 64)
    model = neural_manifold_operator(shape_in=(10, 1, 64, 64))
    output, x_rec = model(inputs)

    print(output.shape, x_rec.shape) 
