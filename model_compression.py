import torch
import numpy as np
from functools import reduce
import os
import glob
import urllib
import zipfile
import collections
import argparse
from torch.nn.utils import prune

from compress import prepare_model, prepare_dataloader, compress_and_save, load_and_decompress, compress_and_decompress
from PIL import Image

parser = argparse.ArgumentParser(description='Model compression')

parser.add_argument('--compress_ratio_encoder', type=float, default=0.2)
parser.add_argument('--compress_ratio_generator', type=float, default=0.2)
parser.add_argument('--compress_ratio_hyperprior', type=float, default=0.2)
cmd_args = parser.parse_args()


MODEL_OUT_DIR = '/scratch/ssd004/scratch/ruizhu/hific/ckpt/'
STAGING_DIR = '/scratch/ssd004/scratch/ruizhu/hific/ckpt/stage'
CKPT_DIR = '/scratch/ssd004/scratch/ruizhu/hific/ckpt'

File = collections.namedtuple('File', ['output_path', 'compressed_path',
                                       'num_bytes', 'bpp'])

_ = [os.makedirs(dir, exist_ok=True) for dir in (CKPT_DIR, STAGING_DIR, MODEL_OUT_DIR)]
#
model_path = os.path.join(CKPT_DIR, 'hific_med.pt')

model, args = prepare_model(model_path, STAGING_DIR)
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].shape)

def get_module_by_name(module,
                       access_string):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)
params_to_prune_encoder = [(get_module_by_name(model, 'Encoder.conv_block1.1'), 'weight'),
                    (get_module_by_name(model, 'Encoder.conv_block2.1'), 'weight'),
                    (get_module_by_name(model, 'Encoder.conv_block3.1'), 'weight'),
                    (get_module_by_name(model, 'Encoder.conv_block4.1'), 'weight'),
                    (get_module_by_name(model, 'Encoder.conv_block5.1'), 'weight'),
                    (get_module_by_name(model, 'Encoder.conv_block_out.1'), 'weight')
                   ]
params_to_prune_generator = [(get_module_by_name(model, 'Generator.conv_block_init.2'), 'weight'),
                    (get_module_by_name(model, 'Generator.upconv_block1.0'), 'weight'),
                    (get_module_by_name(model, 'Generator.upconv_block2.0'), 'weight'),
                    (get_module_by_name(model, 'Generator.upconv_block3.0'), 'weight'),
                    (get_module_by_name(model, 'Generator.upconv_block4.0'), 'weight'),
                    (get_module_by_name(model, 'Generator.conv_block_out.1'), 'weight')
                   ]
params_to_prune_hyperprior = [(get_module_by_name(model, 'Hyperprior.analysis_net.conv1'), 'weight'),
                    (get_module_by_name(model, 'Hyperprior.analysis_net.conv2'), 'weight'),
                    (get_module_by_name(model, 'Hyperprior.analysis_net.conv3'), 'weight'),
                    (get_module_by_name(model, 'Hyperprior.synthesis_mu.conv1'), 'weight'),
                    (get_module_by_name(model, 'Hyperprior.synthesis_mu.conv2'), 'weight'),
                    (get_module_by_name(model, 'Hyperprior.synthesis_mu.conv3'), 'weight'),
                    (get_module_by_name(model, 'Hyperprior.synthesis_std.conv1'), 'weight'),
                    (get_module_by_name(model, 'Hyperprior.synthesis_std.conv2'), 'weight'),
                    (get_module_by_name(model, 'Hyperprior.synthesis_std.conv3'), 'weight'),]



print("Model's parameters to prune:")

compress_ratio_encoder = cmd_args.compress_ratio_encoder
print('compress_ratio_encoder', compress_ratio_encoder)
compress_ratio_encoder_str = str(compress_ratio_encoder).replace('.', 'dot')

compress_ratio_generator = cmd_args.compress_ratio_generator
print('compress_ratio_generator', compress_ratio_generator)
compress_ratio_generator_str = str(compress_ratio_generator).replace('.', 'dot')

compress_ratio_hyperprior = cmd_args.compress_ratio_hyperprior
print('compress_ratio_hyperprior', compress_ratio_hyperprior)

compress_ratio_hyperprior_str = str(compress_ratio_hyperprior).replace('.', 'dot')


if compress_ratio_encoder:
    prune.global_unstructured(
        params_to_prune_encoder,
        pruning_method=prune.L1Unstructured,
        amount=compress_ratio_encoder,
    )
    for param in params_to_prune_encoder:
        prune.remove(param[0], param[1])

if compress_ratio_generator:
    prune.global_unstructured(
        params_to_prune_generator,
        pruning_method=prune.L1Unstructured,
        amount=compress_ratio_generator,
    )
    for param in params_to_prune_generator:
        prune.remove(param[0], param[1])
if compress_ratio_hyperprior:
    prune.global_unstructured(
        params_to_prune_hyperprior,
        pruning_method=prune.L1Unstructured,
        amount=compress_ratio_hyperprior,
    )
    for param in params_to_prune_hyperprior:
        prune.remove(param[0], param[1])






torch.save(model, os.path.join(MODEL_OUT_DIR, f"compressed_encoder{compress_ratio_encoder_str}_generator{compress_ratio_generator_str}_hyperprior{compress_ratio_hyperprior}_hific_med.pt"))