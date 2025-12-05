import torch
import torch.nn as nn
from transformers import Swinv2Config, Swinv2Model
from .modules import Swinv2Fields
from .backbones.HAT import HAT
from .utils import LayerNorm2D, grid_mapping
from .model import FIPERModel
from einops import repeat

class FIPER(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Default Configurations (Hardcoded for simplicity as per request)
        self.frequencies = torch.tensor([1, 4, 16, 64], dtype=torch.float32).to(self.device)
        self.scale = 4
        
        # 1. Basis Field
        SWIN_PRETRAINED = "microsoft/swinv2-base-patch4-window8-256"
        pretrained_model = Swinv2Model.from_pretrained(SWIN_PRETRAINED)
        # We need a proper config here. Using the pretrained config as base.
        # In a real scenario, we might want to load a specific config file or params.
        # For this release, we assume standard config matches the reference.
        basis_config = pretrained_model.config
        # Adjust config if necessary (e.g. input size) - checking reference Swinv2Fields usage
        # The reference initialized Swinv2Fields with `swin=Swinv2Model(basis_config)`.
        
        self.basis_field = Swinv2Fields(swin=Swinv2Model(basis_config), freq_bands_num=6, upsample_scalar=4)
        
        # 2. Mat
        hidden_dim = sum(self.basis_field.out_hidden_sizes) * (len(self.frequencies) * 2 + 1)
        freq_dim = self.basis_field.freq_bands_num * (len(self.frequencies) * 2 + 1)
        
        self.mat = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, 
                      out_channels=freq_dim * 4, 
                      kernel_size=7, padding=3, 
                      groups=freq_dim),
            LayerNorm2D(normalized_shape=freq_dim * 4),
            nn.Conv2d(in_channels=freq_dim * 4, 
                      out_channels=3, kernel_size=3, padding=1)
        )
        
        # 3. SwinIR (HAT)
        self.swinir = HAT(
            upscale=1, in_chans=3, out_chans=256, embed_dim=180, 
            depths=[6]*6, num_heads=[6]*6, window_size=16,
            compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
            overlap_ratio=0.5, mlp_ratio=2, drop_path_rate=0.0,
            upsampler='pixelshuffle'
        )
        
        # 4. Coeff Decoder
        self.coeff_decoder = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(64, 256, 3, 1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(64, 256, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(256, hidden_dim, 1, 0)
        )
        
        self.model = FIPERModel(
            swinir=self.swinir,
            basis_field=self.basis_field,
            coeff_decoder=self.coeff_decoder,
            mat=self.mat,
            coeff_basis_channel=sum(self.basis_field.out_hidden_sizes) * len(self.frequencies) * 2,
            frequencies=self.frequencies
        )
        
        self.model.to(self.device)
        self.model.eval()

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
            
    def forward(self, image):
        # Image Preprocessing
        # Assuming image is a PIL Image or Tensor
        # If PIL, convert to tensor. If tensor, ensure shape.
        from torchvision.transforms import functional as TF
        from PIL import Image
        
        if isinstance(image, Image.Image):
             img_tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            img_tensor = image.to(self.device)
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)
        else:
            raise ValueError("Input must be PIL Image or Tensor")
            
        B, C, H, W = img_tensor.shape
        out_H, out_W = H * self.scale, W * self.scale
        
        # Coordinate Generation
        y, x = torch.meshgrid(torch.arange(0, out_H), torch.arange(0, out_W), indexing='ij')
        coords = torch.stack([x, y], dim=-1).float() + 0.5
        
        # Grid Mapping
        # Note: Using out_H/out_W for AABB as standard for super-resolution where entire grid is mapped
        # Assuming frequencies matches the coordinate scale logic
        basis_coords = grid_mapping(
            repeat(coords, 'h w c -> b h w c', b=B).to(self.device),
            aabb=self.frequencies.new([[0, 0], [out_H, out_W]]),
            freq_bands=self.frequencies.new(self.basis_field.freq_bands)
        )
        
        with torch.no_grad():
            output = self.model(img_tensor, basis_coords)
            
        return output

def create_model(ckpt_path=None, device='cuda'):
    model = FIPER(device=device)
    if ckpt_path:
        model.load_checkpoint(ckpt_path)
    return model
