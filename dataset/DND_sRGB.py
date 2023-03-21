from dataset.base import dataset_path
import h5py
import glob
import numpy as np
import os
import scipy.io as sio
from torch.utils.data import Dataset

dnd_path = os.path.join(dataset_path, 'DND')

class DNDSrgbBenchmarkDataset(Dataset):
    def __init__(self):
        super(DNDSrgbBenchmarkDataset, self).__init__()
        self.imgs = []
        infos = h5py.File(os.path.join(dnd_path, 'info.mat'), 'r')
        info = infos['info']
        bb = info['boundingboxes']
        for i in range(50):
            filename = os.path.join(dnd_path, 'images_srgb', '%04d.mat' % (i + 1))
            img = h5py.File(filename, 'r')
            Inoisy = np.float32(np.array(img['InoisySRGB']).T)
            ref = bb[0][i]
            boxes = np.array(info[ref]).T
            for k in range(20):
                idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]
                Inoisy_crop = Inoisy[idx[0]:idx[1], idx[2]:idx[3], :].copy()
                Inoisy_crop = np.transpose(Inoisy_crop, (2, 0, 1)) * 255.

                self.imgs.append({'L':Inoisy_crop})

    def __getitem__(self, index):
        return self.imgs[index]

    def __len__(self):
        return 1000