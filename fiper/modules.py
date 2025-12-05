import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from transformers import Swinv2Model
from copy import deepcopy
from typing import List, Tuple
from .utils import LayerNorm2D, grid_mapping

class Swinv2PatchMerging(nn.Module):

    def __init__(self, input_resolution = None, hidden_size: int = None, dim: int = None, norm_layer = None) -> None:
        
        super().__init__()
        assert hidden_size is not None or dim is not None, "hidden_size or dim should be provided"
        self.hidden_size = hidden_size if hidden_size is not None else dim
        self.reduction = nn.Linear(4 * self.hidden_size, 2 * self.hidden_size, bias = False)
        self.norm = nn.LayerNorm(2 * self.hidden_size) if norm_layer is None else norm_layer(2 * self.hidden_size)
        self.downsampler = nn.PixelUnshuffle(downscale_factor = 2)

    def forward(self, hidden_states: torch.Tensor, input_dimensions = None) -> torch.Tensor:
        
        B, N, C = hidden_states.shape
        assert (N ** 0.5).is_integer(), "N should be a perfect square"
        assert N % 4 == 0, "N should be divisible by 4"
        hidden_states = rearrange(hidden_states, 'b (h w) c -> b c h w', h = int(N ** 0.5), w = int(N ** 0.5), c = self.hidden_size).contiguous()
        # different from original Swin Transformer, we use PixelUnshuffle instead of manual patch merging
        hidden_states = self.downsampler(hidden_states)
        hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c', c = self.hidden_size * 4).contiguous()
        hidden_states = self.reduction(hidden_states)
        hidden_states = self.norm(hidden_states)

        return hidden_states

class Swinv2PatchUnMerging(nn.Module):

    def __init__(self, input_resolution = None, hidden_size: int = None, dim: int = None, norm_layer = None) -> None:
        
        super().__init__()
        assert hidden_size is not None or dim is not None, "hidden_size or dim should be provided"
        self.hidden_size = hidden_size if hidden_size is not None else dim
        self.expansion = nn.Linear(int(self.hidden_size / 4), int(self.hidden_size / 2), bias = False)
        self.norm = nn.LayerNorm(int(self.hidden_size / 2))
        self.upsampler = nn.PixelShuffle(upscale_factor = 2)
    
    def forward(self, hidden_states: torch.Tensor, input_dimensions = None) -> torch.Tensor:

        B, N, C = hidden_states.shape
        assert (N ** 0.5).is_integer(), "N should be a perfect square"
        assert N % 4 == 0, "N should be divisible by 4"
        hidden_states = rearrange(hidden_states, 'b (h w) c -> b c h w', h = int(N ** 0.5), w = int(N ** 0.5), c = self.hidden_size).contiguous()
        hidden_states = self.upsampler(hidden_states)
        hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c', c = int(self.hidden_size / 4)).contiguous()
        hidden_states = self.expansion(hidden_states)
        hidden_states = self.norm(hidden_states)

        return hidden_states
        
    
class FactorFrequencyBandsDownsample(nn.Module):

    def __init__(self, hidden_size: int, output_hidden_size: int = None) -> None:
        
        super().__init__()
        self.hidden_size = hidden_size
        self.output_hidden_size = output_hidden_size if output_hidden_size is not None else 2 * hidden_size
        self.reduction = nn.Linear(4 * self.hidden_size, self.output_hidden_size, bias = False)
        self.norm = nn.LayerNorm(self.output_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        B, N, C = hidden_states.shape
        assert (N ** 0.5).is_integer(), "N should be a perfect square"
        assert N % 4 == 0, "N should be divisible by 4"
        hidden_states = rearrange(hidden_states, 'b (h w) c -> b c h w', h = int(N ** 0.5), w = int(N ** 0.5)).contiguous()
        hidden_states = rearrange(hidden_states, 'b c (e h) (g w) -> b (h w) (e g c)', e = 2, g = 2).contiguous()
        hidden_states = self.reduction(hidden_states)
        hidden_states = self.norm(hidden_states)

        return hidden_states
    

class Swinv2Fields(nn.Module):

    def __init__(self, swin: Swinv2Model, freq_bands_num: int = 6, upsample_scalar: int = 4, out_dim: int = None):
                                                                                    
        super().__init__()
        self.freq_bands_num = freq_bands_num
        self.freq_bands = [2 ** i for i in range(freq_bands_num)]
        # print(f"Frequency bands: {self.freq_bands}")
        assert len(swin.config.depths) == 1, "Swin should have only one layer"
        self.config = swin.config
        assert np.log2(self.config.image_size // self.config.patch_size) % 1 == 0, "Image size divided by patch size should be power of 2"
        self.upsample_scalar = upsample_scalar
        self.output_sizes = [self.upsample_scalar * (2 ** i) for i in range(int(np.log2(self.config.image_size // self.config.patch_size)), int(np.log2(self.config.image_size // self.config.patch_size)) - self.freq_bands_num, -1)]
        # print(f"Output sizes: {self.output_sizes}")
        self.embeddings = swin.embeddings
        self.layers = []
        assert len(swin.encoder.layers) == 1, "Swin should have only one layer"
        for _ in self.freq_bands:
            self.layers.append(deepcopy(swin.encoder.layers[0]))
        self.Layers = nn.ModuleList(self.layers)
        self.downsamplers = nn.ModuleList([FactorFrequencyBandsDownsample(hidden_size = layer.dim, output_hidden_size = layer.dim) for layer in self.Layers[:-1]])
        # print(f"Layers Parameters: {sum(p.numel() for p in self.Layers.parameters())}")
        self.basis_hidden_sizes = [layer.dim for layer in self.layers]
        self.out_hidden_sizes = [(layer.dim // (self.upsample_scalar ** 2) if out_dim is None else out_dim) for layer in self.layers]
        self.basis_decoders = nn.ModuleList([nn.Sequential(
                                                nn.Conv2d(in_channels = layer.dim, out_channels = out_hidden_size * (self.upsample_scalar ** 2), kernel_size = 1, padding = 0, stride = 1),
                                                nn.PixelShuffle(upscale_factor = self.upsample_scalar),
                                                nn.Conv2d(in_channels = out_hidden_size, out_channels = out_hidden_size, kernel_size = 1, padding = 0, stride = 1),
                                            ) for layer, out_hidden_size in zip(self.layers, self.out_hidden_sizes)])
        # print(f"Output hidden sizes: {self.out_hidden_sizes}")
        
    def forward(self, imgs: torch.Tensor = None, hidden_states: torch.Tensor = None, debug: bool = False, return_intermediate = False) -> List[torch.Tensor]:
        
        assert imgs is None or hidden_states is None, "imgs and hidden_states should not be provided at the same time"
        if hidden_states is None:
            B, C, H, W = imgs.shape
            hidden_states, input_dimensions = self.embeddings(imgs)
        else:
            B, Ch, Hh, Wh = hidden_states.shape
            input_dimensions = (Hh, Wh)
            hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c').contiguous()
        assert input_dimensions[0] == input_dimensions[1] and input_dimensions[0] == self.config.image_size // self.config.patch_size, f"Input dimensions should be equal to {self.config.image_size // self.config.patch_size}, but got {input_dimensions[0]} and {input_dimensions[1]}"
        intermediate_basises = []
        basises = []
        for i, layer in enumerate(self.layers):
            if debug:
                print(f"\nMain Layer {i}, Hidden states shape: {hidden_states.shape}")
            for  j, block in enumerate(layer.blocks):
                if debug:
                    print(f"Main Layer {i}, Block {j}, Hidden states shape: {hidden_states.shape}")
                hidden_states = block(hidden_states, input_dimensions)[0]
            intermediate_basises.append(hidden_states)
            basises.append(self.basis_decoders[i](rearrange(hidden_states, 'b (h w) c -> b c h w', h = input_dimensions[0], w = input_dimensions[1]).contiguous()))
            height, width = input_dimensions
            assert height == width and height == self.output_sizes[i] // self.upsample_scalar, f"Height and width should be equal to {self.output_sizes[i] // self.upsample_scalar}, but got {height} and {width}"
            if i < len(self.layers) - 1:
                if debug:
                    assert isinstance(layer.downsample, FactorFrequencyBandsDownsample), "Downsample should be FactorFrequencyBandsDownsample"
                    print(f"Main Layer {i}, Downsample, Hidden states shape: {hidden_states.shape}")
                hidden_states = self.downsamplers[i](hidden_states)
                input_dimensions = (int(input_dimensions[0]) // 2, int(input_dimensions[1]) // 2)
        if return_intermediate:
            return intermediate_basises, basises
        return basises
