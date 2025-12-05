# FIPER: Factorized Features for Robust Image Super-Resolution and Compression (NeurIPS 2025)

[Yang-Che Sun](https://github.com/jayisaking)¹, Cheng Yu Yeo¹, [Ernie Chu](https://www.cs.jhu.edu/~schu23/)², [Jun-Cheng Chen](https://homepage.citi.sinica.edu.tw/pages/pullpull/index_en.html)³, [Yu-Lun Liu](https://yulunalexliu.github.io/)¹

¹National Yang Ming Chiao Tung University, ²Johns Hopkins University, ³Academia Sinica

---

### Abstract
In this work, we propose using a unified representation, termed **Factorized Features**, for low-level vision tasks, where we test on **Single Image Super-Resolution (SISR)** and **Image Compression**. Motivated by the shared principles between these tasks, they require recovering and preserving fine image details, whether by enhancing resolution for SISR or reconstructing compressed data for Image Compression. Unlike previous methods that mainly focus on network architecture, our proposed approach utilizes a basis-coefficient decomposition as well as an explicit formulation of frequencies to capture structural components and multi-scale visual features in images, which addresses the core challenges of both tasks. We replace the representation of prior models from simple feature maps with Factorized Features to validate the potential for broad generalizability. In addition, we further optimize the compression pipeline by leveraging the mergeable-basis property of our Factorized Features, which consolidates shared structures on multi-frame compression. Extensive experiments show that our unified representation delivers state-of-the-art performance, achieving an average relative improvement of 204.4% in PSNR over the baseline in Super-Resolution (SR) and 9.35% BD-rate reduction in Image Compression compared to the previous SOTA.

**TL;DR**: An unified image representation to reconstruct the fine details.

---

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

## Citation

```bibtex
@inproceedings{sun2025fiper,
title={{FIPER}: Factorized Features for Robust Image Super-Resolution and Compression},
author={Yang-Che Sun and Cheng Yu Yeo and Ernie Chu and Jun-Cheng Chen and Yu-Lun Liu},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=gcrDTxZTl0}
}
```
