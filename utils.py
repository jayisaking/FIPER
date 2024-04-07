import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from typing import List
import os
import pickle
from glob import glob
import torchvision.transforms as T
import numpy as np
from einops import repeat


def grid_mapping(positions, freq_bands = torch.tensor([2., 4., 6., 8., 10., 12.]), aabb = torch.tensor([[0, 0], [224, 224]])):
    '''
    sawtooth mapping
    '''
    aabbSize = max(aabb[1] - aabb[0])
    scale = aabbSize[..., None] / freq_bands
    pts_local = (positions - aabb[0])[..., None] % scale
    pts_local = pts_local / (scale / 2) - 1
    # pts_local = pts_local.clamp(-1., 1.)
    assert 1 >= pts_local.abs().max(), f'pts_local should be in the range of [-1, 1], but got {pts_local.abs().max()}'
    
    return pts_local

def normalize_coord(xyz, aabb = torch.tensor([[0, 0, 0], [224, 224, 16]])):

    assert (aabb[1] >= aabb[0]).all()
    invaabbSize = 2.0 / (aabb[1] - aabb[0])
    return (xyz - aabb[0]) * invaabbSize - 1

def load_checkpoint(model, ckpt_state_dict_raw):
    '''
    load weights that have the same shape and key in both model and ckpt
    '''
    try:
        model_dict = model.state_dict()
        ckpt_state_dict = {k: v for k, v in ckpt_state_dict_raw.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(ckpt_state_dict)
        model.load_state_dict(model_dict)
        print(f'The following keys is in ckpt but not loaded: {set(ckpt_state_dict_raw.keys()) - set(ckpt_state_dict.keys())}')
    except Exception as e:
        print(e)
    finally:
        return model

def collate_fn(batch):
    
    imgs = torch.stack([item[0] for item in batch])
    imgs_normalized = torch.stack([item[1] for item in batch])
    coordinates = torch.stack([item[2] for item in batch])
    imgs_original = torch.stack([item[3] for item in batch])
    # coordinates_original = torch.stack([item[4] for item in batch])
    H, W = imgs.shape[2:]
    H_max, W_max = imgs_original.shape[2:]
    B = len(batch)
    assert imgs.shape == imgs_normalized.shape == (B, 3, H, W)
    assert H == W and H_max == W_max, f'H: {H}, W: {W}, H_max: {H_max}, W_max: {W_max}'
    interpolate_edge = np.random.randint(H, H_max + 1)
    imgs_original = T.functional.resize(imgs_original, size = (interpolate_edge, interpolate_edge))
    if not F.mse_loss(T.functional.resize(imgs_original, size = (H, W)), imgs) < 1e-3:
        interpolate_edge = H
        imgs_original = imgs
        
    y, x = torch.meshgrid(torch.arange(0, interpolate_edge), torch.arange(0, interpolate_edge), indexing = 'ij')
    coords_original = torch.stack([x, y], dim = -1) + 0.5 # H, W, 2
    coords_original = repeat(coords_original, 'h w c -> b h w c', b = B)
    
    assert F.mse_loss(F.grid_sample(imgs_original, normalize_coord(coords_original, aabb = coords_original.new([[0, 0], [interpolate_edge, interpolate_edge]])), align_corners = False), imgs_original) <= 1e-4
    return imgs, imgs_normalized, coordinates, imgs_original, coords_original



class ImageSetDataset(Dataset):

    def __init__(self, root_dir: List, preprocess, transform, normalize):

        self.root_dir = root_dir
        self.preprocess = preprocess
        self.transform = transform
        self.normalize = normalize
        self.imgs = []
        for dir in self.root_dir:
            if os.path.exists(os.path.join(dir, 'paths.pkl')):
                print('Found paths.pkl, loading images from it.')
                with open(os.path.join(dir, 'paths.pkl'), 'rb') as f:
                    self.imgs += pickle.load(f)
            else:
                print('Loading images from', dir)
                imgs = glob(os.path.join(dir, '*.jpeg'))
                with open(os.path.join(dir, 'paths.pkl'), 'wb') as f:
                    print(f'Saving paths.pkl to { dir }')
                pickle.dump(imgs, f)
                self.imgs += imgs
        print('Found', len(self.imgs), 'images')

        self.H, self.W = self.transform(self.load_image(self.imgs[0])).shape[1:]
        y, x = torch.meshgrid(torch.arange(0, self.H), torch.arange(0, self.W), indexing = 'ij')
        self.coordinates_2D = torch.stack([x, y], dim = -1).float() + 0.5 # H, W, 2
        assert self.coordinates_2D.shape == (self.H, self.W, 2)

        self.original_H, self.original_W = self.load_image(self.imgs[0]).shape[1:]
        y, x = torch.meshgrid(torch.arange(0, self.original_H), torch.arange(0, self.original_W), indexing = 'ij')
        self.original_coordinates_2D = torch.stack([x, y], dim = -1).float() + 0.5 # H, W, 2
        assert self.original_coordinates_2D.shape == (self.original_H, self.original_W, 2)

        print(f"Dataset with {len(self.imgs)} images of shape HW = {self.H}x{self.W} and original HW = {self.original_H}x{self.original_W}")
        print(f"Preprocess: { preprocess }, transform: { transform }, normalize: { normalize }")

    def __len__(self):

        return len(self.imgs)

    def load_image(self, path):
            
        img = Image.open(path).convert('RGB')
        img = self.preprocess(img)
    
        return img
        
    def __getitem__(self, idx):
        
        img_original = self.load_image(self.imgs[idx])
        img = self.transform(img_original)
        img_normalized = self.normalize(img)
        coordinate = self.coordinates_2D
        coordinate_original = self.original_coordinates_2D

        assert (F.grid_sample(img[None], normalize_coord(coordinate[None], aabb = torch.tensor([[0, 0], [self.W, self.H]])), align_corners = False) - img).abs().max() < 1e-3, (F.grid_sample(img[None], normalize_coord(coordinate[None], aabb = torch.tensor([[0, 0], [self.W, self.H]])), align_corners = True) - img).abs().max()
        assert (F.grid_sample(img_original[None], normalize_coord(coordinate_original[None], aabb = torch.tensor([[0, 0], [self.original_W, self.original_H]])), align_corners = False) - img_original).abs().max() < 1e-3, (F.grid_sample(img_original[None], normalize_coord(coordinate_original[None], aabb = torch.tensor([[0, 0], [self.original_W, self.original_H]])), align_corners = True) - img_original).abs().max()
        assert img.shape == (3, self.H, self.W) and img_original.shape == (3, self.original_H, self.original_W)

        return img, img_normalized, coordinate, img_original, coordinate_original
    