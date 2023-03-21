import sys
sys.path.append('..')
import argparse
from skimage.metrics import peak_signal_noise_ratio
import torch
from torch.utils.data import DataLoader
from utils.option import parse, recursive_print
from utils.build import build

def validate_sidd(model, sidd_loader):
    psnrs, count = 0, 0
    for data in sidd_loader:
        output = model.validation_step(data)
        output = torch.floor(output + 0.5)
        output = torch.clamp(output, 0, 255)
        output = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
        gt = data['H'].squeeze(0).permute(1, 2, 0).numpy()
        psnr = peak_signal_noise_ratio(output, gt, data_range=255)
        psnrs += psnr
        count += 1
    return psnrs / count


def main(opt):
    validation_loaders = []
    for validation_dataset_opt in opt['validation_datasets']:
        ValidationDataset = getattr(__import__('dataset'), validation_dataset_opt['type'])
        validation_set = build(ValidationDataset, validation_dataset_opt['args'])
        validation_loader = DataLoader(validation_set, batch_size=1)
        validation_loaders.append(validation_loader)

    Model = getattr(__import__('model'), opt['model'])
    model = Model(opt)
    model.data_parallel()
    if 'resume_from' in opt:
        model.load_model(opt['resume_from'])

    for validation_loader in validation_loaders:
        psnr = validate_sidd(model, validation_loader)
        print('%s, psnr: %6.4f' % (validation_loader.dataset.__class__.__name__, psnr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate the denoiser")
    parser.add_argument("--config_file", type=str, default='../option/three_stage.json')
    argspar = parser.parse_args()

    opt = parse(argspar.config_file)
    recursive_print(opt)

    main(opt)