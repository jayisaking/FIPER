import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .modules import Swinv2Fields
from .backbones.HAT import HAT
from .utils import grid_mapping, LayerNorm2D

class FIPERModel(nn.Module):
    def __init__(self, swinir, basis_field, coeff_decoder, mat, coeff_basis_channel, frequencies):
        
        super(FIPERModel, self).__init__()
        self.swinir = swinir
        self.basis_field = basis_field
        self.coeff_decoder = coeff_decoder
        self.mat = mat
        self.frequencies = frequencies
        self.coeff_basis_channel = coeff_basis_channel

    def forward(self, imgs, basis_coords):

        coeff = self.swinir(imgs)
        # assert coeff.shape[2:] == imgs.shape[2:], f'coeff should have shape {imgs.shape}, but got {coeff.shape}'
        basises = self.basis_field(coeff)

        basis_list = []
                                                                        
        for idx, fi in enumerate(basises):
            # Grid sample the basis fields
            basis = F.grid_sample(fi, grid = basis_coords[..., idx], align_corners = False, mode = 'bicubic', padding_mode = 'border')
            basis_list.append(basis)

        basises = torch.cat(basis_list, dim = 1)
        
        freq_basis = basises[..., None] * self.frequencies
        
        # Decode coefficients and combine with frequency basis
        feat = self.coeff_decoder(coeff) * torch.cat(((rearrange(torch.cat([torch.sin(freq_basis), torch.cos(freq_basis)], dim = -1), 'b c h w f -> b (f c) h w', f = len(self.frequencies) * 2).contiguous()), torch.ones_like(basises, device = basises.device).float()), dim = 1)
        rgb = self.mat(feat)
    
        return rgb
