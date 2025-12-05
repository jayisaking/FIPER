# FIPER: Frequency-based Image Restoration

This repository contains the inference code for FIPER.

## Installation

```bash
pip install torch torchvision transformers einops pillow numpy
```


## Usage
```python
from PIL import Image
from torchvision.transforms import functional as TF
from fiper import create_model

# Initialize Model
model = create_model(ckpt_path='path/to/checkpoint.pth', device='cuda')

# Load Image
img = Image.open('test.png').convert('RGB')

# Inference
result = model(img)

# Save
TF.to_pil_image(result.squeeze(0).clamp(0, 1).cpu()).save("output.png")
```

## TODO
- [ ] Image Compression
- [ ] More Coefficient Backbones
- [ ] Training Code
