# FIPER: Frequency-based Image Restoration

This repository contains the inference code for FIPER.

## Code Structure

- `FIPER/fiper/`: Core package containing the model, modules, and backbones.
- `FIPER/README.md`: This file.

## Usage

Easy inference using the simplified wrapper:

```python
from PIL import Image
from torchvision.transforms import functional as TF
from fiper import create_model

# 1. Initialize Model
# Automatically constructs the architecture and optionally loads checkpoint
model = create_model(ckpt_path='path/to/checkpoint.pth', device='cuda')

# 2. Load Image
image_path = "path/to/image.png"
image = Image.open(image_path).convert('RGB')

# 3. Inference
# The model handles preprocessing (to tensor) and coordinate generation internally
result = model(image)

# 4. Save Result
# Result is a tensor (B, C, H, W)
output_pil = TF.to_pil_image(result.squeeze(0).clamp(0, 1).cpu())
output_pil.save("output.png")
```

### Advanced Usage

If you need to customize individual components, you can import them directly from `FIPER.fiper.modules`, `FIPER.fiper.backbones`, etc.
