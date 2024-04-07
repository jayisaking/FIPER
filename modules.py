import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import numpy as np
from copy import deepcopy
from typing import List
from transformers import Swinv2Model

class Swinv2PatchMerging(nn.Module):

    def __init__(self, hidden_size: int) -> None:
        
        super().__init__()
        self.hidden_size = hidden_size
        self.reduction = nn.Linear(4 * self.hidden_size, 2 * self.hidden_size, bias = False)
        self.norm = nn.LayerNorm(2 * self.hidden_size)
        self.downsampler = nn.PixelUnshuffle(downscale_factor = 2)

    def forward(self, hidden_states: torch.Tensor, input_dimenssions = None) -> torch.Tensor:
        
        B, N, C = hidden_states.shape
        assert (N ** 0.5).is_integer(), "N should be a perfect square"
        assert N % 4 == 0, "N should be divisible by 4"
        hidden_states = rearrange(hidden_states, 'b (h w) c -> b c h w', h = int(N ** 0.5), w = int(N ** 0.5))
        # different from original Swin Transformer, we use PixelUnshuffle instead of manual patch merging
        hidden_states = self.downsampler(hidden_states)
        hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c', c = self.hidden_size * 4)
        hidden_states = self.reduction(hidden_states)
        hidden_states = self.norm(hidden_states)

        return hidden_states

class Swinv2PatchUnMerging(nn.Module):

    def __init__(self, hidden_size: int) -> None:
        
        super().__init__()
        self.hidden_size = hidden_size
        self.expansion = nn.Linear(int(self.hidden_size / 4), int(self.hidden_size / 2), bias = False)
        self.norm = nn.LayerNorm(int(self.hidden_size / 2))
        self.upsampler = nn.PixelShuffle(upscale_factor = 2)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        B, N, C = hidden_states.shape
        assert (N ** 0.5).is_integer(), "N should be a perfect square"
        assert N % 4 == 0, "N should be divisible by 4"
        hidden_states = rearrange(hidden_states, 'b (h w) c -> b c h w', h = int(N ** 0.5), w = int(N ** 0.5))
        hidden_states = self.upsampler(hidden_states)
        hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c', c = int(self.hidden_size / 4))
        hidden_states = self.expansion(hidden_states)
        hidden_states = self.norm(hidden_states)

        return hidden_states
        

    
class FactorFrequencyBandsDownsample(nn.Module):

    def __init__(self, hidden_size: int) -> None:
        
        super().__init__()
        self.hidden_size = hidden_size
        self.reduction = nn.Linear(4 * self.hidden_size, 2 * self.hidden_size, bias=False)
        self.norm = nn.LayerNorm(2 * self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        B, N, C = hidden_states.shape
        assert (N ** 0.5).is_integer(), "N should be a perfect square"
        assert N % 4 == 0, "N should be divisible by 4"
        hidden_states = rearrange(hidden_states, 'b (h w) c -> b c h w', h = int(N ** 0.5), w = int(N ** 0.5))
        hidden_states = rearrange(hidden_states, 'b c (e h) (g w) -> b (h w) (e g c)', e = 2, g = 2)
        hidden_states = self.reduction(hidden_states)
        hidden_states = self.norm(hidden_states)

        return hidden_states
    

class Swinv2Fields(nn.Module):

    def __init__(self, swin: Swinv2Model, freq_bands_num: int = 6, freq_sizes: List[int] = [8, 8, 8, 8, 4, 2], upsample_scalar: int = 8, output_mlp_ratio: int = 2):

        super().__init__()
        assert freq_bands_num == len(freq_sizes), "freq_bands_num should be equal to len(freq_sizes)"
        self.freq_bands_num = freq_bands_num
        self.freq_bands = [2 ** i for i in range(freq_bands_num)]
        self.freq_sizes = freq_sizes
        self.upsample_scalar = upsample_scalar
        assert len(swin.config.depths) == len(self.freq_bands), "Swin depth should be equal to freq_bands_num"
        self.config = swin.config
        assert np.log2(self.config.image_size // self.config.patch_size) % 1 == 0, "Image size divided by patch size should be power of 2"
        self.output_sizes = [2 ** i for i in range(int(np.log2(self.config.image_size // self.config.patch_size)), int(np.log2(self.config.image_size // self.config.patch_size)) - len(self.config.depths) , -1)]
        self.embeddings = swin.embeddings
        self.main_layers = swin.encoder.layers
        for layer in self.main_layers:
            if layer.downsample is not None:
                layer.downsample = FactorFrequencyBandsDownsample(hidden_size = layer.dim)
        auxiliary_layers = []
        for idx, (output_size, target_size) in enumerate(zip(self.output_sizes, self.freq_sizes)):
            auxiliary_layers.append(nn.ModuleList([]))
            if output_size != target_size:
                auxiliary_layers[-1].append(Swinv2PatchMerging(hidden_size = self.main_layers[idx].dim))
                auxiliary_num = int(np.log2(output_size // target_size))
                for layer_idx in range(idx + 1, idx + 1 + auxiliary_num):
                    auxiliary_layers[-1].append(deepcopy(self.main_layers[layer_idx]))
                    auxiliary_layers[-1][-1].blocks = auxiliary_layers[-1][-1].blocks[:2]
                    auxiliary_layers[-1][-1].downsample = Swinv2PatchMerging(hidden_size = self.main_layers[layer_idx].dim) if layer_idx != (idx + 1 + auxiliary_num - 1) else None
            else:
                auxiliary_layers[-1].append(nn.Identity())

        self.auxiliary_layers = nn.ModuleList(auxiliary_layers)

        self.basis_hidden_size = [self.main_layers[self.output_sizes.index(freq_size)].dim for freq_size in self.freq_sizes]
        assert all([size % (self.upsample_scalar ** 2) == 0 for size in self.basis_hidden_size]), "Basis hidden size should be divisible by upsample_scalar ** 2"   
        self.basis_layernorms = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in self.basis_hidden_size])
        self.output_mlp_ratio = output_mlp_ratio
        self.basis_output_hidden_size = [int(dim // (self.upsample_scalar ** 2)) * self.output_mlp_ratio for dim in self.basis_hidden_size]
        self.basis_decoders = nn.ModuleList([nn.Conv2d(in_channels = hidden_size // self.output_mlp_ratio, out_channels = hidden_size, kernel_size = 1, padding = 0) for hidden_size in self.basis_output_hidden_size])
        print(f"Frequency bands: {self.freq_bands}, Frequency sizes: {self.freq_sizes}, Output sizes: {self.output_sizes}, Basis hidden sizes: {self.basis_hidden_size}, Basis output hidden sizes: {self.basis_output_hidden_size}")

    def forward(self, imgs: torch.Tensor, debug: bool = False) -> List[torch.Tensor]:

        B, C, H, W = imgs.shape
        hidden_states, input_dimenssions = self.embeddings(imgs)
        basises = []
        for i, layer in enumerate(self.main_layers):
            if debug:
                print(f"\nMain Layer {i}, Hidden states shape: {hidden_states.shape}")
            for  j, block in enumerate(layer.blocks):
                if debug:
                    print(f"Main Layer {i}, Block {j}, Hidden states shape: {hidden_states.shape}")
                hidden_states = block(hidden_states, input_dimenssions)[0]
            auxiliary_hidden_states = hidden_states
            height, width = input_dimenssions
            assert height == width and height == self.output_sizes[i], "Height and width should be equal to output_sizes"
            if len(self.auxiliary_layers[i]) > 1:
                auxiliary_hidden_states = self.auxiliary_layers[i][0](auxiliary_hidden_states)
                input_dimenssions_auxiliary = (int(input_dimenssions[0]) // 2, int(input_dimenssions[1]) // 2)
                for ia, auxiliary_layer in enumerate(self.auxiliary_layers[i][1:]):
                    for ja, block in enumerate(auxiliary_layer.blocks):
                        auxiliary_hidden_states = block(auxiliary_hidden_states, input_dimenssions_auxiliary)[0]
                        if debug:
                            print(f"Main Layer {i}, Auxiliary Layer {ia}, Block {ja}, Hidden states shape: {auxiliary_hidden_states.shape}")
                    if auxiliary_layer.downsample is not None:
                        auxiliary_hidden_states = auxiliary_layer.downsample(auxiliary_hidden_states)
                        if debug:
                            assert isinstance(auxiliary_layer.downsample, Swinv2PatchMerging), "Downsample should be Swinv2PatchMerging"
                            print(f"Main Layer {i}, Auxiliary Layer {ia}, Downsample, Hidden states shape: {auxiliary_hidden_states.shape}")
                        input_dimenssions_auxiliary = (int(input_dimenssions_auxiliary[0]) // 2, int(input_dimenssions_auxiliary[1]) // 2)

            assert auxiliary_hidden_states.shape[:-1] == (B, self.freq_sizes[i] ** 2), f"Auxiliary hidden states shape should be equal to (B, {self.freq_sizes[i] ** 2}), but got {auxiliary_hidden_states.shape[:-1]}"
            basises.append(self.basis_layernorms[i](auxiliary_hidden_states))

            if layer.downsample is not None:
                if debug:
                    assert isinstance(layer.downsample, FactorFrequencyBandsDownsample), "Downsample should be FactorFrequencyBandsDownsample"
                    print(f"Main Layer {i}, Downsample, Hidden states shape: {hidden_states.shape}")
                hidden_states = layer.downsample(hidden_states)
                input_dimenssions = (int(input_dimenssions[0]) // 2, int(input_dimenssions[1]) // 2)
        assert all([basis.shape == (B, self.freq_sizes[i] ** 2, self.basis_hidden_size[i]) for i, basis in enumerate(basises)]), "All basis should have the same shape"
        basises_output = [self.basis_decoders[i](F.pixel_shuffle(rearrange(basis, 'b (h w) c -> b c h w', h = self.freq_sizes[i], w = self.freq_sizes[i]), self.upsample_scalar)) for i, basis in enumerate(basises)]

        return basises, basises_output
        


class LayerNorm2D(nn.Module):
    '''
    ConvNeXt-Style LayerNorm2D (!= GroupNorm with group = 1)
    '''
    def __init__(self, normalized_shape, eps = 1e-6, group = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.group = group
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b (g c) h w -> (b g) c h w', g = self.group)
        u = x.mean(1, keepdim = True)
        s = (x - u).pow(2).mean(1, keepdim = True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = rearrange(x, '(b g) c h w -> b (g c) h w', g = self.group)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x