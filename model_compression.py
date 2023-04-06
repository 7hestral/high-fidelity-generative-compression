import torch
import numpy as np

import os
import glob
import urllib
import zipfile
import collections

from torch.nn.utils import prune

from compress import prepare_model, prepare_dataloader, compress_and_save, load_and_decompress, compress_and_decompress
from PIL import Image

MODEL_OUT_DIR = './ckpt/out'
STAGING_DIR = './content/stage'
CKPT_DIR = './ckpt'

File = collections.namedtuple('File', ['output_path', 'compressed_path',
                                       'num_bytes', 'bpp'])

_ = [os.makedirs(dir, exist_ok=True) for dir in (CKPT_DIR, STAGING_DIR)]
#
model_path = './ckpt/hific_med.pt'

model, args = prepare_model(model_path, STAGING_DIR)
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].shape)
parameters_to_prune = [
    (module, "weight") for module in filter(lambda m: type(m) == torch.nn.Conv2d, model.modules())
]
print("Model's parameters to prune:")
print(parameters_to_prune)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

for param in parameters_to_prune:
    prune.remove(param[0], param[1])

torch.save(model, os.path.join(MODEL_OUT_DIR, "compressed_hific_med.pt"))