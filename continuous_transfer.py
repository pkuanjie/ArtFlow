import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from pathlib import Path


import time
import numpy as np
import random


def test_transform(img, size):
    transform_list = []
    h, w, _ = np.shape(img)
    if h<w:
        newh = size
        neww = w/h*size
    else:
        neww = size
        newh = h/w*size
    neww = int(neww//4*4)
    newh = int(newh//4*4)
    transform_list.append(transforms.Resize((newh, neww)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str, default='input/content/golden_gate.jpg',
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str, default='input/content',
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str, default='input/style/la_muse.jpg',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str, default='input/style',
                    help='Directory path to a batch of style images')
parser.add_argument('--decoder', type=str, default='experiments/decoder2.pth.tar')

# Additional options
parser.add_argument('--size', type=int, default=256,
                    help='New size for the content and style images, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# glow parameters
parser.add_argument('--operator', type=str, default='adain',
                    help='style feature transfer operator')
parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')# 32
parser.add_argument('--n_block', default=2, type=int, help='number of blocks')# 4
parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.operator == 'wct':
    from glow_wct import Glow
elif args.operator == 'adain':
    from glow_adain import Glow
elif args.operator == 'decorator':
    from glow_decorator import Glow
else:
    raise('Not implemented operator', args.operator)


output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

assert (args.content or args.content_dir)
assert (args.style or args.style_dir)

content_dir = Path(args.content_dir)
content_paths = [f for f in content_dir.glob('*')]
style_dir = Path(args.style_dir)
style_paths = [f for f in style_dir.glob('*')]



# glow
glow = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)

# -----------------------resume training------------------------
if os.path.isfile(args.decoder):
    print("--------loading checkpoint----------")
    checkpoint = torch.load(args.decoder)
    args.start_iter = checkpoint['iter']
    glow.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'".format(args.decoder))
else:
    print("--------no checkpoint found---------")
glow = glow.to(device)
glow.eval()

# -----------------------start------------------------
for content_path in content_paths:
    count = 0
    for style_path in style_paths[1:21]:
        if count==0:
            content = Image.open(str(content_path)).convert('RGB')
            img_transform = test_transform(content, args.size)
            content = img_transform(content)
            content = content.to(device).unsqueeze(0)
        else:
            content = output.to(device)

        style = Image.open(str(style_path)).convert('RGB')
        img_transform = test_transform(style, args.size)
        style = img_transform(style)
        style = style.to(device).unsqueeze(0)
        
        with torch.no_grad():
            # content/style ---> z ---> stylized 
            z_c = glow(content, forward=True)
            z_s = glow(style, forward=True)
            output = glow(z_c, forward=False, style=z_s)
            output = output.cpu()

        count += 1
        output_name = output_dir / '{:s}_count_{:s}_stylized_{:s}{:s}'.format(
            content_path.stem, str(count), style_path.stem, args.save_ext)
        save_image(output, str(output_name))
        print(output_name)
        
        next_path = output_name
            
