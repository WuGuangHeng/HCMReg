# import math
import collections
import numpy as np
from skimage.transform import resize
import random

class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False):
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            shape = im.shape[1:dim+1]
            self.sample(*shape)

        if isinstance(img, collections.abc.Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)]

        return self.tf(img)

    def __str__(self):
        return 'Identity()'

class MinMax_norm(Base):
    def __init__(self, ):
        a = None

    def tf(self, img, k=0):
        if k == 1:
            return img
        img = (img - img.min()) / (img.max()-img.min())
        return img

class Seg_norm(Base):
    def __init__(self, dataset='LPBA40'):
        a = None
        if dataset == 'LPBA40':
            self.seg_table = np.array([0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
                                       63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
                                       163, 164, 165, 166, 181, 182])
        elif dataset == 'MindBoggle':
            self.seg_table = np.array([0, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 
                                       1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035, 2002, 2003, 2005, 2006, 
                                       2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 
                                       2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035])  # Mindboggle
        elif dataset == 'OASIS':
            self.seg_table = np.array([i for i in range(0, 36)])
        elif dataset == 'IXI':
            self.seg_table = np.array([0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                                       28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                                       63, 72, 77, 80, 85, 251, 252, 253, 254, 255])
        elif dataset == 'SR_Reg' or dataset == 'SR-Reg_data':
            self.seg_table = np.array([0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28])

    def tf(self, img, k=0):
        if k == 0:
            return img
        img_out = np.zeros_like(img)
        for i in range(len(self.seg_table)):
            img_out[img == self.seg_table[i]] = i
        return img_out


class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types  # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)


class Resize_img(Base):
    def __init__(self, shape):
        self.shape = shape

    def tf(self, img, k=0):
        if k == 1:
            img = resize(
                img,
                (img.shape[0], self.shape[0], self.shape[1], self.shape[2]),
                anti_aliasing=False,
                order=0,
            )
        else:
            img = resize(
                img,
                (img.shape[0], self.shape[0], self.shape[1], self.shape[2]),
                anti_aliasing=False,
                order=3,
            )
        return img


class Flip(Base):
    def __init__(self, axis=0):
        self.axis = axis

    def tf(self, img, k=0):
        return np.flip(img, self.axis)

    def __str__(self):
        return "Flip(axis={})".format(self.axis)


class RandomFlip(Base):
    # mirror flip across all x,y,z
    def __init__(self, axis=0):
        # assert axis == (1,2,3) # For both data and label, it has to specify the axis.
        self.axis = (1, 2, 3)
        self.x_buffer = None
        self.y_buffer = None
        self.z_buffer = None

    def sample(self, *shape):
        self.x_buffer = np.random.choice([True, False])
        self.y_buffer = np.random.choice([True, False])
        self.z_buffer = np.random.choice([True, False])
        return list(shape)  # the shape is not changed

    def tf(self, img, k=0):  # img shape is (1, 240, 240, 155, 4)
        if self.x_buffer:
            img = np.flip(img, axis=self.axis[0])
        if self.y_buffer:
            img = np.flip(img, axis=self.axis[1])
        if self.z_buffer:
            img = np.flip(img, axis=self.axis[2])
        return img
    
class CenterCrop(Base):
    def __init__(self, size):
        self.size = size
        self.buffer = None

    def sample(self, *shape):
        size = self.size
        start = [(s - size) // 2 for s in shape]
        self.buffer = [slice(None)] + [slice(s, s + size) for s in start]
        return [size] * len(shape)

    def tf(self, img, k=0):
        # print(img.shape)#(1, 240, 240, 155, 4)
        return img[tuple(self.buffer)]
        # return img[self.buffer]

    def __str__(self):
        return "CenterCrop({})".format(self.size)


class CenterCropBySize(CenterCrop):
    def sample(self, *shape):
        assert len(self.size) == 3  # random crop [H,W,T] from img [240,240,155]
        if not isinstance(self.size, list):
            size = list(self.size)
        else:
            size = self.size
        start = [(s - i) // 2 for i, s in zip(size, shape)]
        self.buffer = [slice(None)] + [slice(s, s + i) for i, s in zip(size, start)]
        return size

    def __str__(self):
        return "CenterCropBySize({})".format(self.size)


class RandCrop(CenterCrop):
    def sample(self, *shape):
        size = self.size
        start = [random.randint(0, s - size) for s in shape]
        self.buffer = [slice(None)] + [slice(s, s + size) for s in start]
        return [size] * len(shape)

    def __str__(self):
        return "RandCrop({})".format(self.size)


class RandCrop3D(CenterCrop):
    def sample(self, *shape):  # shape : [240,240,155]
        assert len(self.size) == 3  # random crop [H,W,T] from img [240,240,155]
        if not isinstance(self.size, list):
            size = list(self.size)
        else:
            size = self.size
        start = [random.randint(0, s - i) for i, s in zip(size, shape)]
        self.buffer = [slice(None)] + [slice(s, s + k) for s, k in zip(start, size)]
        return size

    def __str__(self):
        return "RandCrop({})".format(self.size)