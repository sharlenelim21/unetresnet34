"""
models/cyclegan.py — CycleGAN for grayscale cardiac MRI domain translation
===========================================================================
Translates ACDC scanner images (domain A) to rv_landmark scanner style (domain B).

Architecture:
  Generator  : ResNet-9 (9 residual blocks), InstanceNorm, 1-channel I/O
  Discriminator: PatchGAN 70x70, InstanceNorm

Loss:
  Adversarial : LSGAN  (MSE, more stable than BCE)
  Cycle       : L1, lambda_cycle=10
  Identity    : L1, lambda_identity=5

Both generator and discriminator expect images normalised to [-1, 1].
"""

import torch
import torch.nn as nn


# ── building blocks ───────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    ResNet-9 generator.
    in_channels / out_channels = 1 (grayscale MRI).
    ngf = base number of filters (default 64).
    """

    def __init__(self, in_channels=1, out_channels=1, ngf=64, n_res_blocks=9):
        super().__init__()

        # initial convolution
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # downsampling (×2 twice → 64x64)
        for mult in [1, 2]:
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]

        # residual blocks at bottleneck resolution
        for _ in range(n_res_blocks):
            layers.append(_ResBlock(ngf * 4))

        # upsampling back to 256×256
        for mult in [4, 2]:
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2,
                                   3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]

        # final convolution → tanh maps to [-1, 1]
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """
    PatchGAN 70×70 discriminator.
    ndf = base number of filters (default 64).
    """

    def __init__(self, in_channels=1, ndf=64, n_layers=3):
        super().__init__()

        # first layer — no InstanceNorm
        layers = [
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        mult = 1
        for i in range(1, n_layers):
            prev_mult = mult
            mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(ndf * prev_mult, ndf * mult, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf * mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        prev_mult = mult
        mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * prev_mult, ndf * mult, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # output patch map (no sigmoid — LSGAN uses raw logits)
        layers.append(nn.Conv2d(ndf * mult, 1, 4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)


# ── image buffer for training stability ──────────────────────────────────────

class ImageBuffer:
    """
    Stores a rolling buffer of 50 previously generated images.
    Returns a random mix of fresh and buffered images to the discriminator,
    reducing oscillation (Shrivastava et al., 2017).
    """

    def __init__(self, max_size=50):
        self.max_size = max_size
        self.buffer = []

    def push_and_pop(self, images):
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(img)
                result.append(img)
            else:
                if torch.rand(1).item() > 0.5:
                    idx = torch.randint(0, len(self.buffer), (1,)).item()
                    old = self.buffer[idx].clone()
                    self.buffer[idx] = img
                    result.append(old)
                else:
                    result.append(img)
        return torch.cat(result, dim=0)
