from torch import nn
import torch
import ai8x

from kp2dsonar.utils.image import image_grid
from kp2dsonar.utils.logging import timing
"""
Network description class
"""
class ai84_keypointnet(nn.Module):
    """
    7-Layer CNN - Lightweight image classification
    """
    def __init__(self, n_features=256, num_channels=3, bias=True, device = "cuda", **kwargs):
        super().__init__()
        ai8x.set_device(84, None, False)
        self.device = device

        c1, c2, c3, c4, c5, d1 = 32, 64, 128, 256, 256, 512

        self.conv1a = ai8x.FusedConv2dBNReLU(in_channels = num_channels, out_channels = c1, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        self.conv1b = ai8x.FusedConv2dBNReLU(in_channels = c1, out_channels = c1, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)        
        self.conv2a = ai8x.FusedMaxPoolConv2dBNReLU(in_channels = c1, out_channels = c2, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        self.conv2b = ai8x.FusedConv2dBNReLU(in_channels = c2, out_channels = c2, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        self.conv3a = ai8x.FusedConv2dBNReLU(in_channels = c2, out_channels = c3, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        self.conv3b = ai8x.FusedMaxPoolConv2dBNReLU(in_channels = c3, out_channels = c3, kernel_size = 3,
                                    padding=1, bias=bias, **kwargs)

        self.convDa = ai8x.FusedMaxPoolConv2dBNReLU(in_channels = c3, out_channels = c4, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        self.convDb = ai8x.Conv2d(in_channels = c4, out_channels = 1, kernel_size = 3,
                                    padding=1, bias=bias, **kwargs)


        self.convPa = ai8x.FusedMaxPoolConv2dBNReLU(in_channels = c3, out_channels = c4, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        self.convPb = ai8x.Conv2d(in_channels = c4, out_channels = 2, kernel_size = 3,
                                    padding=1, bias=bias, **kwargs)


        self.convFa = ai8x.FusedConv2dBNReLU(in_channels = c3, out_channels = c4, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        self.convFb = ai8x.FusedConv2dBNReLU(in_channels = c4, out_channels = c4, kernel_size = 3,
                                    padding=1, bias=bias, **kwargs)

        self.convFaa = ai8x.FusedMaxPoolConv2dBNReLU(in_channels = c4, out_channels = c5, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        self.convFbb = ai8x.Conv2d(in_channels = c5, out_channels = n_features, kernel_size = 3,
                                    padding=1, bias=bias, **kwargs)
        self.cell = 8
        self.bn_momentum = 0.1
        self.cross_ratio = 2.0
    """
    Assemble the model
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # # Data plotting - for debug
        # matplotlib.use('MacOSX')
        # plt.imshow(x[1, 0], cmap="gray")
        # plt.show()
        # breakpoint()

        B, _, H, W = x.shape
        
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        
        score = self.convDa(x)
        score = self.convDb(score) #removed sigmoid because not supported
        score = torch.sigmoid(score)
        B, _, Hc, Wc = score.shape

        border_mask = torch.ones(B, Hc, Wc)
        border_mask[:, 0] = 0
        border_mask[:, Hc - 1] = 0
        border_mask[:, :, 0] = 0
        border_mask[:, :, Wc - 1] = 0
        border_mask = border_mask.unsqueeze(1)
        score = score * border_mask.to(score.device)
        
        center_shift = self.convPa(x)
        center_shift = self.convPb(center_shift)
        center_shift = torch.tanh(center_shift)

        step = (self.cell-1) / 2.
        center_base = image_grid(B, Hc, Wc,
                                 dtype=center_shift.dtype,
                                 device=center_shift.device,
                                 ones=False, normalized=False).mul(self.cell) + step

        coord_un = center_base.add(center_shift.mul(self.cross_ratio * step))
        coord = coord_un.clone()

        coord[:, 0] = torch.clamp(coord_un[:, 0], min=0, max=W-1)
        coord[:, 1] = torch.clamp(coord_un[:, 1], min=0, max=H-1)

        feat = self.convFa(x)
        feat = self.convFb(feat)
        feat = self.convFaa(feat)
        feat = self.convFbb(feat)

        return score, coord, feat
