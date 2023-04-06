import torch
import numpy as np

import os
import glob
import urllib
import zipfile
import collections
from compress import prepare_model, prepare_dataloader, compress_and_save, load_and_decompress, compress_and_decompress
from PIL import Image

INPUT_DIR = './content/files'
STAGING_DIR = './content/stage'
OUT_DIR = './content/out'
CKPT_DIR = './ckpt'
DEFAULT_IMAGE_PREFIX = ('https://storage.googleapis.com/hific/clic2020/images/originals/')

File = collections.namedtuple('File', ['output_path', 'compressed_path',
                                       'num_bytes', 'bpp'])

_ = [os.makedirs(dir, exist_ok=True) for dir in (INPUT_DIR, STAGING_DIR, OUT_DIR,
                                                 CKPT_DIR)]
original_sizes = dict()

def get_default_image(output_dir, image_choice="portrait"):
    image_ID = dict(cafe="b1b8f33917a40c9d0b118ef801de67d4.png",
                    cat="4fa92b8ecb4ee46a942837447de1ac5c.png",
                    city="b98ec5b29d02ef65e57d23ef90660b4d.png",
                    clocktower="9cbf2594f339c0d3d0f0ea25c62af52b.png",
                    fresco="8181526d9f238726d3e1d3ec3cc56fb7.png",
                    islet="c6658d87c608b631f5cc3fb5a8d89731.png",
                    mountain="d3688a7285d7b2b81febe1cd72e6e22c.png",
                    pasta="f5be5054c01d8efc834d78a991356ad6.png",
                    pines="e903c4f4684100a6dbac1f0b9b4de760.png",
                    plaza="d78b363974ac79908b79012f48de715d.png",
                    portrait="ad249bba099568403dc6b97bc37f8d74.png",
                    shoreline="b9bad0c68eb9ce94e02e9698c8cc429a.png",
                    street="90b622e11ecc37edd42297427403ee81.png",
                    tundra="cc831c904a314a0e98530124526e930b.png",
                    )[image_choice]

    default_image_url = os.path.join(DEFAULT_IMAGE_PREFIX, image_ID)
    output_path = os.path.join(output_dir, os.path.basename(default_image_url))
    print('Downloading', default_image_url, '\n->', output_path)
    urllib.request.urlretrieve(default_image_url, output_path)
#
model_path = './ckpt/hific_med.pt'
first_model_init = False

default_image = "portrait"

get_default_image(INPUT_DIR, default_image)

all_files = os.listdir(INPUT_DIR)
print(f'Got following files ({len(all_files)}):')
scale_factor = 2 if len(all_files) == 1 else 4
print(torch.cuda.is_available())
model, args = prepare_model(model_path, STAGING_DIR)
data_loader = prepare_dataloader(args, INPUT_DIR, OUT_DIR)
compress_and_save(model, args, data_loader, OUT_DIR)
all_outputs = []


def get_bpp(image_dimensions, num_bytes):
    w, h = image_dimensions
    return num_bytes * 8 / (w * h)

for compressed_file in glob.glob(os.path.join(OUT_DIR, '*.hfc')):
    file_name, _ = os.path.splitext(compressed_file)
    # output_path = os.path.join(OUT_DIR, f'{file_name}.png')
    output_path = f'{file_name}.png'
    print(output_path)
    # Model decode
    reconstruction = load_and_decompress(model, compressed_file, output_path)

    all_outputs.append(File(output_path=output_path,
                            compressed_path=compressed_file,
                            num_bytes=os.path.getsize(compressed_file),
                            bpp=get_bpp(Image.open(output_path).size, os.path.getsize(compressed_file))))

torch.cuda.empty_cache()