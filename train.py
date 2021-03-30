import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from torchvision.utils import save_image
import time

import net
from sampler import InfiniteSamplerWrapper

from math import log, sqrt, pi

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=4)

parser.add_argument('--mse_weight', type=float, default=0)
parser.add_argument('--style_weight', type=float, default=1)
parser.add_argument('--content_weight', type=float, default=0.1)

# save options
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--print_interval', type=int, default=100)
parser.add_argument('--save_model_interval', type=int, default=5000)
parser.add_argument('--start_iter', type=int, default=0, help='starting iteration')
parser.add_argument('--resume', default='glow.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# glow parameters
parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')# 32
parser.add_argument('--n_block', default=2, type=int, help='number of blocks')# 4
parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')
parser.add_argument('--operator', type=str, default='adain',
                    help='style feature transfer operator')

args = parser.parse_args()

if args.operator == 'wct':
    from glow_wct import Glow
elif args.operator == 'adain':
    from glow_adain import Glow
elif args.operator == 'decorator':
    from glow_decorator import Glow
else:
    raise('Not implemented operator', args.operator)
    
device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
args.resume = os.path.join(args.save_dir, args.resume)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

# VGG
vgg = net.vgg
vgg.load_state_dict(torch.load(args.vgg))
encoder = net.Net(vgg)
encoder = nn.DataParallel(encoder)
encoder.to(device)

# glow
glow_single = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)

# l1 loss
mseloss = nn.MSELoss()

# -----------------------resume training------------------------
if args.resume:
    if os.path.isfile(args.resume):
        print("--------loading checkpoint----------")
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_iter = checkpoint['iter']
        glow_single.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("--------no checkpoint found---------")
glow_single = glow_single.to(device)
glow = nn.DataParallel(glow_single)
glow.train()



# -------------------------------------------------------------
content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(glow.module.parameters(), lr=args.lr)
if args.resume:
    if os.path.isfile(args.resume):
        optimizer.load_state_dict(checkpoint['optimizer'])

log_c = []
log_s = []
log_mse = []
Time = time.time()
# -----------------------training------------------------
for i in range(args.start_iter, args.max_iter):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)

    # glow forward: real -> z_real, style -> z_style
    if i == args.start_iter:
        with torch.no_grad():
            _ = glow.module(content_images, forward=True)
            continue

    # (log_p, logdet, z_outs) = glow()
    z_c = glow(content_images, forward=True)
    z_s = glow(style_images, forward=True)
    # reverse 
    stylized = glow(z_c, forward=False, style=z_s)

    loss_c, loss_s = encoder(content_images, style_images, stylized)
    loss_c = loss_c.mean()
    loss_s = loss_s.mean()
    loss_mse = mseloss(content_images, stylized)
    loss_style = args.content_weight*loss_c + args.style_weight*loss_s + args.mse_weight*loss_mse

    # optimizer update
    optimizer.zero_grad()
    loss_style.backward()
    nn.utils.clip_grad_norm(glow.module.parameters(), 5)
    optimizer.step()
    
    # update loss log
    log_c.append(loss_c.item())
    log_s.append(loss_s.item())
    log_mse.append(loss_mse.item())

    # save image
    if i % args.print_interval == 0:
        with torch.no_grad():
            # stylized ---> z ---> content
            z_stylized = glow(stylized, forward=True)
            real = glow(z_stylized, forward=False, style=z_c)
            
            # pick another content
            another_content = next(content_iter).to(device)

            # stylized ---> z ---> another real
            z_ac = glow(another_content, forward=True)
            another_real = glow(z_stylized, forward=False, style=z_ac)

        output_name = os.path.join(args.save_dir, "%06d.jpg" % i)
        output_images = torch.cat((content_images.cpu(), style_images.cpu(), stylized.cpu(), 
                                    real.cpu(), another_content.cpu(), another_real.cpu()), 
                                  0)
        save_image(output_images, output_name, nrow=args.batch_size)
        
        print("iter %d   time/iter: %.2f   loss_c: %.3f   loss_s: %.3f   loss_mse: %.3f" % (i, 
                                                                      (time.time()-Time)/args.print_interval, 
                                                                      np.mean(np.array(log_c)), np.mean(np.array(log_s)),
                                                                      np.mean(np.array(log_mse))
                                                                       ))
        log_c = []
        log_s = []
        Time = time.time()

        
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = glow.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))

        state = {'iter': i, 'state_dict': state_dict, 'optimizer': optimizer.state_dict()}
        torch.save(state, args.resume)

