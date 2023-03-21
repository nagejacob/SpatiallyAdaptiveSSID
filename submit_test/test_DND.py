import sys
sys.path.append('..')
import argparse
from dataset.DND_sRGB import dnd_path
import h5py
import numpy as np
import os
import scipy.io as sio
import shutil
from submit_test.ensemble_wrapper import EnsembleWrapper
import torch
from tqdm import tqdm
from utils.option import parse, recursive_print

def bundle_submissions_srgb(submission_folder):
    '''
    Bundles submission data for sRGB denoising

    submission_folder Folder where denoised images reside

    Output is written to <submission_folder>/bundled/. Please submit
    the content of this folder.
    '''
    out_folder = os.path.join(submission_folder, "bundled/")
    try:
        os.mkdir(out_folder)
    except:
        pass
    israw = False
    eval_version = "1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=np.object)
        for bb in range(20):
            filename = '%04d_%02d.mat' % (i + 1, bb + 1)
            s = sio.loadmat(os.path.join(submission_folder, filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat' % (i + 1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )

max_margin = 80

def main(opt):
    if opt['save_mat']:
        if os.path.exists(opt['mat_dir']):
            shutil.rmtree(opt['mat_dir'])
        os.makedirs(opt['mat_dir'])

    Model = getattr(__import__('model'), opt['model'])
    model = Model(opt)
    model.data_parallel()
    if 'resume_from' in opt:
        model.load_model(opt['resume_from'])
    if opt['ensemble']:
        model = EnsembleWrapper(model)

    infos = h5py.File(os.path.join(dnd_path, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    for i in tqdm(range(50)):
        filename = os.path.join(dnd_path, 'images_srgb', '%04d.mat' % (i + 1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)
        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(20):
            idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]

            # Crop margin for better boundary process
            h_min_margin = max_margin
            h_max_margin = max_margin
            w_min_margin = max_margin
            w_max_margin = max_margin

            if 0 > idx[0] - max_margin:
                h_min_margin = idx[0]
            if Inoisy.shape[0] < idx[1] + max_margin:
                h_max_margin = Inoisy.shape[0] - idx[1]
            if 0 > idx[2] - max_margin:
                w_min_margin = idx[2]
            if Inoisy.shape[1] < idx[3] + max_margin:
                w_max_margin = Inoisy.shape[1] - idx[3]

            h_min_margin = h_min_margin // 32 * 32
            h_max_margin = h_max_margin // 32 * 32
            w_min_margin = w_min_margin // 32 * 32
            w_max_margin = w_max_margin // 32 * 32

            Inoisy_crop = Inoisy[idx[0] - h_min_margin:idx[1] + h_max_margin,
                          idx[2] - w_min_margin:idx[3] + w_max_margin, :].copy()
            H = Inoisy_crop.shape[0]
            W = Inoisy_crop.shape[1]

            Inoisy_crop = torch.from_numpy(Inoisy_crop).permute(2, 0, 1).unsqueeze(0).cuda()
            Inoisy_crop = Inoisy_crop * 255.

            Idenoised_crop = model.validation_step({'L': Inoisy_crop})

            Idenoised_crop = torch.clamp(Idenoised_crop / 255., 0., 1.)
            Idenoised_crop = Idenoised_crop.permute(0, 2, 3, 1)[:, h_min_margin:H-h_max_margin, w_min_margin:W-w_max_margin, :].cpu()
            Idenoised_crop = Idenoised_crop.numpy()


            if opt['save_mat']:
                # save denoised data
                Idenoised_crop = np.float32(Idenoised_crop)
                save_file = os.path.join(opt['mat_dir'], '%04d_%02d.mat' % (i + 1, k + 1))
                sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the denoiser")
    parser.add_argument("--config_file", type=str, default='../option/three_stage.json')
    argspar = parser.parse_args()

    opt = parse(argspar.config_file)
    opt['mat_dir'] = 'dnd_mat'
    opt['save_mat'] = True
    opt['ensemble'] = True
    recursive_print(opt)

    main(opt)
    if opt['save_mat']:
        bundle_submissions_srgb(opt['mat_dir'])