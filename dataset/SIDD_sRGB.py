'''
We observe slightly better performance with training inputs in [0, 255] range than that in [0, 1],
so we follow AP-BSN that do not normalize the input image from [0, 255] to [0, 1].
'''
from dataset.base import BaseTrainDataset, dataset_path
import glob
import numpy as np
import os
from PIL import Image
import scipy.io as sio
from torch.utils.data import Dataset

sidd_path = os.path.join(dataset_path, 'SIDD')

class SIDDSrgbTrainDataset(BaseTrainDataset):
    def __init__(self, patch_size, pin_memory):
        super(SIDDSrgbTrainDataset, self).__init__(sidd_path, patch_size, pin_memory)

    def __getitem__(self, index):
        index = index % len(self.img_paths)

        if self.pin_memory:
            img_L = self.imgs[index]['L']
            img_H = self.imgs[index]['H']
        else:
            img_path = self.img_paths[index]
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])

        img_L, img_H = self.crop(img_L, img_H)
        img_L, img_H = self.augment(img_L, img_H)

        img_L, img_H = np.float32(img_L), np.float32(img_H)
        return {'L': img_L, 'H': img_H}

    def _get_img_paths(self, path):
        self.img_paths = []
        L_pattern = os.path.join(path, 'SIDD_Medium_Srgb/Data/*/*_NOISY_SRGB_*.PNG')
        L_paths = sorted(glob.glob(L_pattern))
        for L_path in L_paths:
            self.img_paths.append({'L': L_path, 'H': L_path.replace('NOISY', 'GT')})

    def _open_images(self):
        self.imgs = []
        for img_path in self.img_paths:
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])
            self.imgs.append({'L': img_L, 'H': img_H})

    def _open_image(self, path):
        img = Image.open(path)
        img = np.asarray(img)
        img = np.transpose(img, (2, 0, 1))
        return img


class SIDDSrgbValidationDataset(Dataset):
    def __init__(self):
        super(SIDDSrgbValidationDataset, self).__init__()
        self._open_images(sidd_path)
        self.n = self.noisy_block.shape[0]
        self.k = self.noisy_block.shape[1]

    def __getitem__(self, index):
        index_n = index // self.k
        index_k = index % self.k

        img_H = self.gt_block[index_n, index_k]
        img_H = np.float32(img_H)
        img_H = np.transpose(img_H, (2, 0, 1))

        img_L = self.noisy_block[index_n, index_k]
        img_L = np.float32(img_L)
        img_L = np.transpose(img_L, (2, 0, 1))

        return {'H':img_H, 'L':img_L}

    def __len__(self):
        return self.n * self.k

    def _open_images(self, path):
        mat = sio.loadmat(os.path.join(path, 'SIDD_Validation/ValidationNoisyBlocksSrgb.mat'))
        self.noisy_block = mat['ValidationNoisyBlocksSrgb']
        mat = sio.loadmat(os.path.join(path, 'SIDD_Validation/ValidationGtBlocksSrgb.mat'))
        self.gt_block = mat['ValidationGtBlocksSrgb']


class SIDDSrgbBenchmarkDataset(Dataset):
    def __init__(self):
        super(SIDDSrgbBenchmarkDataset, self).__init__()
        self._open_images(sidd_path)
        self.n = self.noisy_block.shape[0]
        self.k = self.noisy_block.shape[1]

    def __getitem__(self, index):
        index_n = index // self.k
        index_k = index % self.k

        img_L = self.noisy_block[index_n, index_k]
        img_L = np.float32(img_L)
        img_L = np.transpose(img_L, (2, 0, 1))

        return {'L':img_L}

    def __len__(self):
        return self.n * self.k

    def _open_images(self, path):
        mat = sio.loadmat(os.path.join(path, 'SIDD_Benchmark/BenchmarkNoisyBlocksSrgb.mat'))
        self.noisy_block = mat['BenchmarkNoisyBlocksSrgb']