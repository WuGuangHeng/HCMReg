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

def niigzload(fname, label_path, dataset="LPBA40", half_res=False, test=False):
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
    if dataset == 'MM_WHS':
        label_file = os.path.join(label_path, str(
            fname).split('/')[-1].replace('image', 'label'))  # for MM-WHS but not used

    if dataset == 'MM_WHS' and not test:
        label = None
        return sitk.GetArrayFromImage(img), label
    else:
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

        # transforms work with bcdhw
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]

        if self.transforms is not None:
            x, x_seg = self.transforms([x, x_seg])
            y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x) 
        x_seg = np.ascontiguousarray(x_seg).astype(np.int16)
        y = np.ascontiguousarray(y)
        y_seg = np.ascontiguousarray(y_seg).astype(np.int16)
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(x[0, :, :, 8], cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(y[0, :, :, 8], cmap='gray')
        # plt.show()
        # sys.exit(0)
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
       atlas: fixed image  subject: moving image
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
        # transforms work with bcdhw
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        if self.transforms is not None:
            x, x_seg = self.transforms([x, x_seg])
            y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg).astype(np.int16)
        y_seg = np.ascontiguousarray(y_seg).astype(np.int16)
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(x[0, :, :, 8], cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(y[0, :, :, 8], cmap='gray')
        # plt.show()
        # sys.exit(0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x_seg, y_seg = torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y #, x_seg, y_seg # y, x 

    def __len__(self):

        if self.dataset == 'LPBA40' or self.dataset == 'IXI':
            return len(self.paths)
        else:
            return len(self.paths)*(len(self.paths)-1)
        
class BrainInferDatasetA2S(Dataset):
    """Brain dataset for atlas to subject registration
       dataset: LPBA40, IXI
       atlas: fixed image  subject: moving image
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
        return  x, y, x_seg, y_seg # y, x, y_seg, x_seg #  

    def __len__(self):

        if self.dataset == 'LPBA40' or self.dataset == 'IXI':
            return len(self.paths)
        else:
            return len(self.paths)*(len(self.paths)-1)


'''
Half resolution: not implemented yet

class LPBABrainHalfDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def half_pair(self, pair):
        return pair[0][::2, ::2, ::2], pair[1][::2, ::2, ::2]

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        x, x_seg = self.half_pair(pkload(path_x))
        y, y_seg = self.half_pair(pkload(path_y))

        # print(x.shape)
        # print(x.shape)
        # print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x, y = self.transforms([x, y])
        # y = self.one_hot(y, 2)
        # print(y.shape)
        # sys.exit(0)
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(x[0, :, :, 8], cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(y[0, :, :, 8], cmap='gray')
        # plt.show()
        # sys.exit(0)
        # y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)


class LPBABrainHalfInferDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def half_pair(self, pair):
        return pair[0][::2, ::2, ::2], pair[1][::2, ::2, ::2]

    def __getitem__(self, index):
        x_index = index//(len(self.paths)-1)
        s = index % (len(self.paths)-1)
        y_index = s+1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        # print(os.path.basename(path_x), os.path.basename(path_y))
        x, x_seg = self.half_pair(pkload(path_x))
        y, y_seg = self.half_pair(pkload(path_y))
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        # [Bsize,channelsHeight,,Width,Depth]
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(
            y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)
'''

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
    
class SRRegBrainDataset(Dataset):
    def __init__(self, mr_data_path, ct_data_path, transforms):
        self.mr_data_path = mr_data_path
        self.ct_data_path = ct_data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        mr_path = self.mr_data_path[index]
        ct_path = self.ct_data_path[index]

        x, x_seg = pkload(mr_path)
        y, y_seg = pkload(ct_path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x,x_seg= self.transforms([x, x_seg])
        y,y_seg= self.transforms([y, y_seg])
      
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)

        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x_seg, y_seg = torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.ct_data_path)


class SRRegBrainInferDataset(Dataset):
    def __init__(self, mr_data_path, ct_data_path, transforms):
        self.mr_data_path = mr_data_path
        self.ct_data_path = ct_data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        mr_path = self.mr_data_path[index]
        ct_path = self.ct_data_path[index]

        x, x_seg = pkload(mr_path)
        y, y_seg = pkload(ct_path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.ct_data_path)

if __name__ == '__main__':
    train_path = r'/home/wgh/experiments/exps/Proposed_methods/IXI/train'
    test_path = r'/home/wgh/experiments/exps/Proposed_methods/IXI/test'
    label_path = r'/home/wgh/experiments/exps/Proposed_methods/IXI/label'
    atlas_path = r'/home/wgh/experiments/exps/Proposed_methods/IXI/fixed.nii.gz'
  
    test_set = BrainInferDatasetA2S(test_path, label_path, atlas_path, None, dataset='IXI')
    print(len(test_set))
    print(test_set[0][0].shape, test_set[0][1].shape)
