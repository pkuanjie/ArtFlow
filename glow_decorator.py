import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))

import torch
import torch.nn.functional as F

def covsqrt_mean(feature, inverse=False, tolerance=1e-14):
    # I referenced the default svd tolerance value in matlab.

    b, c, h, w = feature.size()
    mean = torch.mean(feature.view(b, c, -1), dim=2, keepdim=True)
    zeromean = feature.view(b, c, -1) - mean
    cov = torch.bmm(zeromean, zeromean.transpose(1, 2))

    evals, evects = torch.symeig(cov, eigenvectors=True)
    
    p = 0.5
    if inverse:
        p *= -1

    covsqrt = []
    for i in range(b):
        k = 0
        for j in range(c):
            if evals[i][j] > tolerance:
                k = j
                break
        covsqrt.append(torch.mm(evects[i][:, k:],
                            torch.mm(evals[i][k:].pow(p).diag_embed(),
                                     evects[i][:, k:].t())).unsqueeze(0))
    covsqrt = torch.cat(covsqrt, dim=0)

    return covsqrt, mean
    

def whitening(feature):
    b, c, h, w = feature.size()
    
    inv_covsqrt, mean = covsqrt_mean(feature, inverse=True)

    normalized_feature = torch.matmul(inv_covsqrt, feature.view(b, c, -1)-mean)
    
    return normalized_feature.view(b, c, h, w)


def coloring(feature, target):
    b, c, h, w = feature.size()

    covsqrt, mean = covsqrt_mean(target)
    
    colored_feature = torch.matmul(covsqrt, feature.view(b, c, -1)) + mean
    
    return colored_feature.view(b, c, h, w)

def extract_patches(feature, patch_size, stride):
    ph, pw = patch_size
    sh, sw = stride
    
    # padding the feature
    padh = (ph - 1) // 2
    padw = (pw - 1) // 2
    padding_size = (padw, padw, padh, padh)
    feature = F.pad(feature, padding_size, 'constant', 0)

    # extract patches
    patches = feature.unfold(2, ph, sh).unfold(3, pw, sw)
    patches = patches.contiguous().view(*patches.size()[:-2], -1)
    
    return patches

class StyleDecorator(torch.nn.Module):
    
    def __init__(self):
        super(StyleDecorator, self).__init__()

    def kernel_normalize(self, kernel, k=3):
        b, ch, h, w, kk = kernel.size()
        
        # calc kernel norm
        kernel = kernel.view(b, ch, h*w, kk).transpose(2, 1)
        kernel_norm = torch.norm(kernel.contiguous().view(b, h*w, ch*kk), p=2, dim=2, keepdim=True)
        
        # kernel reshape
        kernel = kernel.view(b, h*w, ch, k, k)
        kernel_norm = kernel_norm.view(b, h*w, 1, 1, 1)
        
        return kernel, kernel_norm

    def conv2d_with_style_kernels(self, features, kernels, patch_size, deconv_flag=False):
        output = list()
        b, c, h, w = features.size()
        
        # padding
        pad = (patch_size - 1) // 2
        padding_size = (pad, pad, pad, pad)
        
        # batch-wise convolutions with style kernels
        for feature, kernel in zip(features, kernels):
            feature = F.pad(feature.unsqueeze(0), padding_size, 'constant', 0)
                
            if deconv_flag:
                padding_size = patch_size - 1
                output.append(F.conv_transpose2d(feature, kernel, padding=padding_size))
            else:
                output.append(F.conv2d(feature, kernel))
        
        return torch.cat(output, dim=0)
        
    def binarize_patch_score(self, features):
        outputs= list()
        
        # batch-wise operation
        for feature in features:
            matching_indices = torch.argmax(feature, dim=0)
            one_hot_mask = torch.zeros_like(feature)

            h, w = matching_indices.size()
            for i in range(h):
                for j in range(w):
                    ind = matching_indices[i, j]
                    one_hot_mask[ind, i, j] = 1
            outputs.append(one_hot_mask.unsqueeze(0))
            
        return torch.cat(outputs, dim=0)
   
    def norm_deconvolution(self, h, w, patch_size):
        mask = torch.ones((h, w))
        fullmask = torch.zeros((h + patch_size - 1, w + patch_size - 1))

        for i in range(patch_size):
            for j in range(patch_size):
                pad = (i, patch_size - i - 1, j, patch_size - j - 1)
                padded_mask = F.pad(mask, pad, 'constant', 0)
                fullmask += padded_mask

        pad_width = (patch_size - 1) // 2
        if pad_width == 0:
            deconv_norm = fullmask
        else:
            deconv_norm = fullmask[pad_width:-pad_width, pad_width:-pad_width]

        return deconv_norm.view(1, 1, h, w)

    def reassemble_feature(self, normalized_content_feature, normalized_style_feature, patch_size, patch_stride):
        # get patches of style feature
        style_kernel = extract_patches(normalized_style_feature, [patch_size, patch_size], [patch_stride, patch_stride])

        # kernel normalize
        style_kernel, kernel_norm = self.kernel_normalize(style_kernel, patch_size)
        
        # convolution with style kernel(patch wise convolution)
        patch_score = self.conv2d_with_style_kernels(normalized_content_feature, style_kernel/kernel_norm, patch_size)
        
        # binarization
        binarized = self.binarize_patch_score(patch_score)
        
        # deconv norm
        deconv_norm = self.norm_deconvolution(h=binarized.size(2), w=binarized.size(3), patch_size=patch_size)

        # deconvolution
        output = self.conv2d_with_style_kernels(binarized, style_kernel, patch_size, deconv_flag=True)
        
        return output/deconv_norm.type_as(output)

    def forward(self, content_feature, style_feature, style_strength=1.0, patch_size=3, patch_stride=1): 
        # 1-1. content feature projection
        normalized_content_feature = whitening(content_feature)

        # 1-2. style feature projection
        normalized_style_feature = whitening(style_feature)

        # 2. swap content and style features
        reassembled_feature = self.reassemble_feature(normalized_content_feature, normalized_style_feature, patch_size=patch_size, patch_stride=patch_stride)

        # 3. reconstruction feature with style mean and covariance matrix
        stylized_feature = coloring(reassembled_feature, style_feature)

        # 4. content and style interpolation
        result_feature = (1 - style_strength) * content_feature + style_strength * stylized_feature
        
        return result_feature
    
class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
            
        return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super().__init__()
        
        if out_channel is None:
            out_channel = in_channel
        weight = torch.randn(in_channel, out_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape
        out = F.conv2d(input, self.weight)
        return out

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super().__init__()

        if out_channel is None:
            out_channel = in_channel
        weight = np.random.randn(in_channel, out_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        return out

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out

        return torch.cat([in_a, out_b], 1)

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, use_coupling=True, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)
            
        self.use_coupling = use_coupling
        if self.use_coupling:
            self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        input = self.actnorm(input)
        input = self.invconv(input)
        if self.use_coupling:
            input = self.coupling(input)
        return input

    def reverse(self, input):
        if self.use_coupling:
            input = self.coupling.reverse(input)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
        for flow in self.flows:
            out = flow(out)

        return out

    def reverse(self, output, reconstruct=False):
        input = output
        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 4
            
        self.blocks.append(Block(n_channel, n_flow, affine=affine))
        self.decorator = StyleDecorator()
        
    def forward(self, input, forward=True, style=None):
        if forward:
            return self._forward(input, style=style)
        else:
            return self._reverse(input, style=style)

    def _forward(self, input, style=None):
        z = input
        for block in self.blocks:
            z = block(z)
        if style is not None:
            z_wct = self.decorator(z, style)
        return z

    def _reverse(self, z, style=None):
        out = z
        if style is not None:
            out_wct = self.decorator(out, style)
            out = out_wct
        for i, block in enumerate(self.blocks[::-1]):
            out = block.reverse(out)
        return out