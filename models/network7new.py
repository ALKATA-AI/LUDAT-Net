import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import glob
import os

from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from joblib import Parallel, delayed
from torchvision.ops.deform_conv import DeformConv2d
from utils import get_padding_2d, LearnableSigmoid_2d
from pesq import pesq
	
###############################################################################
# Encoder & Decoder
###############################################################################

# --- Deep Separatable Dilated Dense Block (DSDDB) ---
class DS_DDB(nn.Module):
    def __init__(self, h, kernel_size=(3, 3), depth=4):
        super(DS_DDB, self).__init__()
        self.h = h
        self.depth = depth
        # self.Deform_Embedding = Deform_Embedding(in_chans=h.dense_channel, embed_dim=h.dense_channel)
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(h.dense_channel*(i+1), h.dense_channel*(i+1), kernel_size, dilation=(dil, 1),
                          padding=get_padding_2d(kernel_size, dilation=(dil, 1)), groups=h.dense_channel*(i+1), bias=True),
                nn.Conv2d(in_channels=h.dense_channel*(i+1), out_channels=h.dense_channel, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):

        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x

# --- Channel attention Block (CAB) ---
class CAB(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Global Average Pooling
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() # Output between 0 and 1 for weighting
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # Squeeze operation
        y = self.fc(y).view(b, c, 1, 1) # Excitation operation
        return x * y.expand_as(x) # Scale operation (element-wise multiplication)

# --- Channel attention Encoder (CAE) ---
class CAE(nn.Module):
    def __init__(self, h, in_channel):
        super(CAE, self).__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

        self.dense_block = DS_DDB(h, depth=4) # [b, h.dense_channel, ndim_time, h.n_fft//2+1]
        #New improvement: Channel Attention Block
        self.channel_attention = CAB(channel=h.dense_channel, reduction=16)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.channel_attention(x)
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x

# --- Channel Attention Magnitude Decoder (CAMD) ---
class CAMD(nn.Module):
    def __init__(self, h, out_channel=1):
        super(CAMD, self).__init__()
        self.dense_block = DS_DDB(h, depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid_2d(h.n_fft//2+1, beta=h.beta)
        #New improvement: Channel Attention Block
        self.channel_attention = CAB(channel=h.dense_channel, reduction=16)
		
    def forward(self, x):
        x = self.dense_block(x)
        x = self.channel_attention(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x

# --- Channel Attention Phase Decoder (CAPD) ---
class CAPD(nn.Module):
    def __init__(self, h, out_channel=1):
        super(CAPD, self).__init__()
        self.dense_block = DS_DDB(h, depth=4)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        #New improvement: Channel Attention Block
        self.channel_attention = CAB(channel=h.dense_channel, reduction=16)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.channel_attention(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x

###############################################################################
# Classes related to Network
###############################################################################

# --- Helper functions ---

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
      

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

#  --- Convolutional Transposed Relative positional encoding Self-Attention (CTRSA) ---
class CTRSA(nn.Module):
    def __init__(self, dim, num_heads, bias=True, dropout=0.0, max_pos_emb=512):
        super(CTRSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dropout = nn.Dropout(dropout)

        # Linear projections
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.qkv_dwconv = nn.Conv1d(dim * 3, dim * 3, kernel_size=3, 
                                   padding=1, groups=dim * 3, bias=bias)
        
        self.project_out = nn.Linear(dim, dim, bias=bias)
        self.dwc = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=num_heads, bias=bias)

        # Relative positional encoding for audio
        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, num_heads)

    def forward(self, x, mask=None):
        b, n, c = x.shape

        # QKV projections with depthwise conv
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (c d) -> b (c d) n', d=3)
        qkv = self.qkv_dwconv(qkv)
        qkv = rearrange(qkv, 'b (c d) n -> b n (c d)', d=3)
        
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Residual feature map
        v_residual = rearrange(v, 'b n c -> b c n')
        feature_map = self.dwc(v_residual)
        feature_map = rearrange(feature_map, 'b c n -> b n c')

        # Multi-head rearrangement
        q = rearrange(q, 'b n (head c) -> b head n c', head=self.num_heads)
        k = rearrange(k, 'b n (head c) -> b head n c', head=self.num_heads)
        v = rearrange(v, 'b n (head c) -> b head n c', head=self.num_heads)

        # Normalize
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        #  **Relative Positional Bias** 
        seq = torch.arange(n, device=x.device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-self.max_pos_emb, self.max_pos_emb) + self.max_pos_emb
        
        # Get relative positional bias
        rel_pos_bias = self.rel_pos_emb(dist)  # Shape: [n, n, num_heads]
        
        # Rearrange to match attention dimensions
        rel_pos_bias = rearrange(rel_pos_bias, 'i j h -> h i j')  # [num_heads, n, n]
        rel_pos_bias = rel_pos_bias.unsqueeze(0)  # [1, num_heads, n, n]
        
        # Add to attention scores
        attn = attn + rel_pos_bias

        # Apply mask if provided
        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max
            mask = mask.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, n)
            attn = attn.masked_fill(~mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = attn @ v
        out = rearrange(out, 'b head n c -> b n (head c)')
        out = out + feature_map
        
        return self.project_out(out)


# --- Feedforward ---
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
			nn.LayerNorm(dim), #new
            nn.Linear(dim, dim * mult),
            nn.SiLU(), #new1(best)
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# --- Conv module ---
class ConvModule(nn.Module):
    def __init__(
        self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            
            nn.Conv1d(dim, inner_dim * 2, 1),

			nn.GLU(dim=1),

            DepthWiseConv1d(
                inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),

            nn.SiLU(), #new

            nn.Conv1d(inner_dim, dim, 1),

            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# --- Depthwise-Augmented Transformer Block (DAT Block) ---
class DAT_Block(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = CTRSA(
            dim=dim,
            num_heads=heads,
            dropout=attn_dropout,
            max_pos_emb=512
        )
        self.conv = ConvModule(
            dim=dim,
            causal=False,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.ff1(x) + x
        x = self.attn(x, mask=mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x

# --- Time-Frequency DAT (TF-DAT) ---
class TF_DAT(nn.Module):
    def __init__(self, h):
        super(TF_DAT, self).__init__()
        self.h = h
        self.time = DAT_Block(
            dim=h.dense_channel,
            dim_head=h.dense_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
            )
        self.freq = DAT_Block(
            dim=h.dense_channel,
            dim_head=h.dense_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
            )
        self.beta = nn.Parameter(torch.zeros((1, 1, 1, h.dense_channel)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, 1, 1, h.dense_channel)), requires_grad=True)
    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        
        x = self.time(x) + x * self.beta
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)

        x = self.freq(x) + x * self.gamma
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x

# --- Downsampling and Upsampling Helper Classes ---

class DownSample(nn.Module):
    # # Downsampling block using strided Conv2d
    def __init__(self, h, C): 
        super(DownSample, self).__init__()
        # # C is the channel count (h.dense_channel)
        self.down = nn.Sequential(
            # # Halve T and F dimensions using stride 2
            nn.Conv2d(C, C, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(C, affine=True),
            nn.PReLU(C)
        )

    def forward(self, x):
        return self.down(x)

# -----------------------------------------------------

class UpSample(nn.Module):
    # # Upsampling block using ConvTranspose2d
    def __init__(self, h, C):
        super(UpSample, self).__init__()
        # # C is the channel count (h.dense_channel)
        self.up = nn.Sequential(
            # # Double T and F dimensions using ConvTranspose2d with stride 2
            nn.ConvTranspose2d(C, C, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.InstanceNorm2d(C, affine=True),
            nn.PReLU(C)
        )

    def forward(self, x):
        return self.up(x)


###############################################################################
# Main Network Class (DAT_UNet) with One-Level U-Net
###############################################################################

class DAT_UNet(nn.Module):
    def __init__(self, h, in_channel=2):
        super(DAT_UNet, self).__init__()
        
        self.h = h
        C = h.dense_channel # # Channel dimension for internal blocks (e.g., 64)
        
        # # 1. Initial Encoder (Converts [B, 2, T, F] to [B, C, T, F/2])
        self.dense_encoder = CAE(h, in_channel) 
        
        # # 2. Processing Blocks (Time-Frequency Separable Conformer Blocks)
        self.ALSEnet = nn.ModuleList([
            TF_DAT(h), # # Block 0: Encoder Level 1
            TF_DAT(h), # # Block 1: Bottleneck
            TF_DAT(h), # # Block 2: Decoder Level 1
            TF_DAT(h)  # # Block 3: Final Refinement 
        ])
        
        # # 3. Down-sampling and Up-sampling modules 
        self.downsample = DownSample(h, C) 
        self.upsample1 = UpSample(h, C) 
        
        # # 4. Skip Connection Layer (to reduce 2C channels back to C after concatenation)
        self.FFM = nn.Sequential(
            nn.Conv2d(2 * C, C, (1, 1)),     # # Input 2C (u1 + e1), Output C
            nn.InstanceNorm2d(C, affine=True), 
            nn.PReLU(C) 
        )
        
        # 5. Learned Residual Skip Connection (LRSC) [from encoder input to final decoder output]
        # -------------------------------------------------------------------------
        self.LRSC = nn.Sequential(
            nn.Conv2d(C, C, (3, 3), (1, 1), padding=(1, 1)), 
            nn.InstanceNorm2d(C, affine=True),
            nn.PReLU(C),
            nn.Conv2d(C, C, (1, 1))
        )

        # # 6. Decoders (Magnitude Mask and Phase)
        self.mask_decoder = CAMD(h, out_channel=1)
        self.phase_decoder = CAPD(h, out_channel=1)


    def forward(self, noisy_mag, noisy_pha): # # Input shape: [B, F, T] (Magnitude and Phase Spectrograms)
        
        # # --- Input Preparation (Spectrogram) ---
        # # Convert from [B, F, T] to [B, 2, T, F] (2 channels: Mag and Pha)
        noisy_mag = noisy_mag.unsqueeze(1).permute(0, 1, 3, 2) # # [B, 1, T, F]
        noisy_pha = noisy_pha.unsqueeze(1).permute(0, 1, 3, 2) # # [B, 1, T, F]
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # # [B, 2, T, F]
        
        # # Initial Dense Encoding
        x_in = self.dense_encoder(x)  # # [B, C, T, F/2]

        # # -------------------------------------------------#
        # # --- One-Level U-Net Core ---
        # # -------------------------------------------------#
        
        # # 1. Encoder Level 1
        e1 = self.ALSEnet[0](x_in)  # # [B, C, T_e1, F_e1] 
        p1 = self.downsample(e1) # # [B, C, T/2, F/4] 
        
        # # 2. Bottleneck (Level 2)
        e2 = self.ALSEnet[1](p1) # # [B, C, T/2, F/4] 
        
        # # 3. Decoder Level 1
        u1 = self.upsample1(e2) # # [B, C, T_u1, F_u1]
        
        # # **FIX: Cropping to match the size of e1 (resolves torch.cat error)**
        _, _, T_e1, F_e1 = e1.size()
        u1 = u1[:, :, :T_e1, :F_e1] # # Crop u1 to the exact dimensions of e1

        s1 = torch.cat((u1, e1), dim=1)                        # # [B, 2C, T, F/2] 
        s1 = self.FFM(s1)                               
        d1 = self.ALSEnet[2](s1)                                

        # # 4. Final Refinement (TS_BLOCK 3)
        d2 = self.ALSEnet[3](d1)                                
        
        # # --- Apply Learned Residual Skip Connection ---
        x_skip = self.LRSC(x_in)                       # # [B, C, T, F/2]
        d_out = d2 + x_skip                                    # # Element-wise addition
        
        # # --- Decoding and Complex Reconstruction ---
        
        # # Decoders output is [B, 1, T, F/2]
        mask = self.mask_decoder(d_out) 
        phase = self.phase_decoder(d_out)

        # # **FIX: Correct Permute/Squeeze to get [B, F/2, T] and [B, F/2, T, 2]**
        
        # # 1. Apply mask
        denoised_mag_masked = noisy_mag * mask # # [B, 1, T, F/2]
        
        # # 2. Re-arrange to [B, F/2, T] for final mag/pha output and stack
        # # Squeeze channel dim (dim=1) and swap T and F/2
        
        # # [B, 1, T, F/2] -> [B, T, F/2] -> [B, F/2, T]
        denoised_mag = denoised_mag_masked.squeeze(1).permute(0, 2, 1) 
        denoised_pha = phase.squeeze(1).permute(0, 2, 1)              
        
        # # 3. Reconstruct complex spectrogram [B, F/2, T, 2]
        # # This is the expected format for torch.istft via the helper function.
        denoised_com = torch.stack((denoised_mag * torch.cos(denoised_pha),
                                    denoised_mag * torch.sin(denoised_pha)), dim=-1) # # [B, F/2, T, 2]
        
        # # Final outputs match the expected return format: [B, F, T], [B, F, T], [B, F, T, 2]
        return denoised_mag, denoised_pha, denoised_com


###############################################################################

def phase_losses(phase_r, phase_g, h):

    dim_freq = h.n_fft // 2 + 1
    dim_time = phase_r.size(-1)

    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - torch.eye(dim_freq)).to(phase_g.device)
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - torch.eye(dim_time)).to(phase_g.device)
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r-phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r-gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r-iaf_g))

    return ip_loss, gd_loss, iaf_loss


def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def pesq_score(utts_r, utts_g, h):

    pesq_score = Parallel(n_jobs=1)(delayed(eval_pesq)(
                            utts_r[i].squeeze().cpu().numpy(),
                            utts_g[i].squeeze().cpu().numpy(), 
                            h.sampling_rate)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)

    return pesq_score


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        # error can happen due to silent period
        pesq_score = -1

    return pesq_score