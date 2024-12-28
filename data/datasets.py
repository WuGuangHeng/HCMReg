import os
import glob
import torch
import sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pickle
import numpy as np

def half_res(img, seg):
    return img[::2, ::2, ::2], seg[::2, ::2, ::2]

def niigzload(fname, label_path, dataset="LPBA40", half_res=False):
    # using simpleitk
    import SimpleITK as sitk
    # load img and label
    img = sitk.ReadImage(fname)
    if dataset == 'LPBA40':
        label_file = os.path.join(
            label_path, str(fname).split('/')[-1][:3] + '.delineation.structure.label.nii.gz')  # for LPBA40
    if dataset == 'MindBoggle':
        label_file = os.path.join(label_path, str(
            fname).split('/')[-1])  # for MindBoggle
    if dataset == 'OASIS' or dataset == 'IXI':
        label_file = os.path.join(label_path, str(
            fname).split('/')[-1].split('.')[0]+'.label.nii.gz')  # for OASIS
   
    label = sitk.ReadImage(label_file)
    img, label = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(label)
    if half_res:
        img, label = half_res(img, label) # train with half resolution
        
        return img, label

def atlasload(atlas_path, half_res=False):
    # using simpleitk
    import SimpleITK as sitk
    # load img and label
    atlas_label_path = atlas_path[:-7] + '_label.nii.gz'
    img = sitk.ReadImage(atlas_path)
    label = sitk.ReadImage(atlas_label_path)
    atlas = sitk.GetArrayFromImage(img)
    atlas_label = sitk.GetArrayFromImage(label)
    if half_res:
        atlas, atlas_label = half_res(atlas, atlas_label) # train with half resolution

    return atlas, atlas_label

class BrainDatasetS2S(Dataset):
    """Brain dataset for subject to subject registration
       dataset: OASIS, MindBoggle
    """
    
    def __init__(self, data_path, label_path, transforms, dataset="OASIS", half_res=False):
        self.paths = [os.path.join(data_path, p)
                      for p in os.listdir(data_path)]
        self.label_path = label_path
        self.transforms = transforms
        self.dataset = dataset
        self.half_res = half_res

    def __getitem__(self, index):
        if self.dataset == 'MindBoggle':
            x_index = index // (len(self.paths) - 1)
            s = index % (len(self.paths) - 1)
            y_index = s + 1 if s >= x_index else s
        elif self.dataset == 'OASIS':
            x_index = index
            y_index = index+1
        
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        x, x_seg = niigzload(path_x, self.label_path, self.dataset, self.half_res)
        y, y_seg = niigzload(path_y, self.label_path, self.dataset, self.half_res)
        # print(x.shape)
        # print(np.unique(y))

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]

        if self.transforms is not None:
            x, x_seg = self.transforms([x, x_seg])
            y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x) 
        x_seg = np.ascontiguousarray(x_seg).astype(np.int16)
        y = np.ascontiguousarray(y)
        y_seg = np.ascontiguousarray(y_seg).astype(np.int16)
    
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x_seg, y_seg = torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        if self.dataset == 'MindBoggle':
            return len(self.paths)*(len(self.paths)-1)
        elif self.dataset == 'OASIS':
            return len(self.paths)-1

class BrainInferDatasetS2S(Dataset):
    """Brain infer dataset for subject to subject registration
         dataset: OASIS, MindBoggle
    """
    def __init__(self, data_path, label_path, transforms, dataset="OASIS", half_res=False):
        self.paths = [os.path.join(data_path, p)
                      for p in os.listdir(data_path)]
        self.label_path = label_path
        self.transforms = transforms
        self.dataset = dataset
        self.half_res = half_res
        
    def __getitem__(self, index):
        if self.dataset == 'MindBoggle':
            x_index = index//(len(self.paths)-1)
            s = index % (len(self.paths)-1)
            y_index = s+1 if s >= x_index else s
        elif self.dataset == 'OASIS':
            x_index = index
            y_index = index+1
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        x, x_seg = niigzload(path_x, self.label_path, self.dataset, self.half_res)
        y, y_seg = niigzload(path_y, self.label_path, self.dataset, self.half_res)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        if self.transforms is not None:
            x, x_seg = self.transforms([x, x_seg])
            y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x) 
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg).astype(np.int16)
        y_seg = np.ascontiguousarray(y_seg).astype(np.int16)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(
            y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        if self.dataset == 'MindBoggle':
            return len(self.paths)*(len(self.paths)-1)
        elif self.dataset == 'OASIS':
            return len(self.paths)-1

class BrainDatasetA2S(Dataset):
    """Brain dataset for atlas to subject registration
       dataset: LPBA40, IXI
    """
    
    def __init__(self, data_path, label_path, atlas_path, transforms, dataset="LPBA40", half_res=False):
        self.paths = [os.path.join(data_path, p)
                      for p in os.listdir(data_path)]
        self.label_path = label_path
        self.atlas_path = atlas_path
        self.transforms = transforms
        self.dataset = dataset
        self.half_res = half_res

    def __getitem__(self, index):
        if self.dataset == 'LPBA40' or self.dataset == 'IXI':
            y_index = index
        else:
            print('Dataset not supported')

        path_y = self.paths[y_index]
        x, x_seg = atlasload(self.atlas_path,self.half_res)
        y, y_seg = niigzload(path_y, self.label_path, self.dataset, self.half_res)

        # print(x.shape)
        # print(np.unique(y))
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        if self.transforms is not None:
            x, x_seg = self.transforms([x, x_seg])
            y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg).astype(np.int16)
        y_seg = np.ascontiguousarray(y_seg).astype(np.int16)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x_seg, y_seg = torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y 

    def __len__(self):

        if self.dataset == 'LPBA40' or self.dataset == 'IXI':
            return len(self.paths)
        else:
            return len(self.paths)*(len(self.paths)-1)
        
class BrainInferDatasetA2S(Dataset):
    """Brain dataset for atlas to subject registration
       dataset: LPBA40, IXI
    """
    
    def __init__(self, data_path, label_path, atlas_path, transforms, dataset="LPBA40", half_res=False):
        self.paths = [os.path.join(data_path, p)
                      for p in os.listdir(data_path)]
        self.label_path = label_path
        self.atlas_path = atlas_path
        self.transforms = transforms
        self.dataset = dataset
        self.half_res = half_res

    def __getitem__(self, index):
        if self.dataset == 'LPBA40' or self.dataset == 'IXI':
            y_index = index
        else:
            print('Dataset not supported')

        path_y = self.paths[y_index]
        x, x_seg = atlasload(self.atlas_path, self.half_res)
        y, y_seg = niigzload(path_y, self.label_path, self.dataset, self.half_res)
       
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        if self.transforms is not None:
            x, x_seg = self.transforms([x, x_seg])
            y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg).astype(np.int16)
        y_seg = np.ascontiguousarray(y_seg).astype(np.int16)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x_seg, y_seg = torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):

        if self.dataset == 'LPBA40' or self.dataset == 'IXI':
            return len(self.paths)
        else:
            return len(self.paths)*(len(self.paths)-1)

