import datetime
import imageio
import numpy as np
import torch

def date_time():
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d, %H:%M:%S")
    return date_time

def log(log_file, str, also_print=True, with_time=True):
    with open(log_file, 'a+') as F:
        if with_time:
            F.write(date_time() + '  ')
        F.write(str)
    if also_print:
        if with_time:
            print(date_time(), end='  ')
        print(str, end='')

# save numpy image in shape 3xHxW
def np2image(image, image_file):
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0., 1.)
    image = image * 255.
    image = image.astype(np.uint8)
    imageio.imwrite(image_file, image)

# save tensor image in shape 1x3xHxW
def tensor2image(image, image_file):
    image = image.detach().cpu().squeeze(0).numpy()
    np2image(image, image_file)

# return pytorch image in shape 1x3xHxW
def image2tensor(image_file):
    image = imageio.imread(image_file).astype(np.float32) / np.float32(255.0)
    if len(image.shape) == 3:
        image = np.transpose(image, (2, 0, 1))
    elif len(image.shape) == 2:
        image = np.expand_dims(image, 0)
    image = np.asarray(image, dtype=np.float32)
    image = torch.from_numpy(image).unsqueeze(0)
    return image
