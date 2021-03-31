import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

class HyperModule(nn.Module):

    def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16):
        super(HyperModule, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        self.proj1 = nn.Linear(z_dim, self.in_size * self.z_dim)
        self.proj2 = nn.Linear(z_dim, self.out_size * self.f_size * self.f_size)

    def forward(self, z):
        h_in = self.proj1(z)
        h_in = h_in.view(self.in_size, self.z_dim)

        h_final = self.proj2(h_in)
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)
        return kernel



class HyperConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',  # TODO: refine this type
                 hyper_module=None,
                 z_dim=512,
                 base=32
                 ):
        super(HyperConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.z_dim = z_dim
        self.base = base
        assert (hyper_module is not None), "no hyper_module"
        self.hyper_module = hyper_module
        self.layer_embed = nn.ParameterList()
        for i in range(self.out_channels // self.base):
            for j in range(max(1, self.in_channels // self.base)):
                self.layer_embed.append(nn.Parameter(torch.fmod(torch.randn(self.z_dim).cuda(), 2)))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        ww = []
        for i in range(self.out_channels // self.base):
            w = []
            for j in range(self.in_channels // self.base):
                w.append(self.hyper_module(
                    self.layer_embed[i * (max(1, self.in_channels // self.base)) + j]))
            ww.append(torch.cat(w, dim=1))
        h_final = torch.cat(ww, dim=0)
        return F.conv2d(x, h_final, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class HyperBasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, z_dim=512, base=16, hyper = None):
        super().__init__(inplanes, planes, stride, downsample, groups, base_width,
                         dilation, norm_layer)
        self.conv1 = HyperConv2d(inplanes, planes, 3, stride, 1, z_dim=z_dim, base=base, hyper_module = hyper)
        self.conv2 = HyperConv2d(planes, planes, 3, 1, 1, z_dim=z_dim, base=base, hyper_module = hyper)


class HyperResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, z_dim=512, base=16):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # HyperNet Stuff.
        self.z_dim = z_dim
        self.base = base

        self.hyper = HyperModule(z_dim= self.z_dim, out_size= self.base, in_size= self.base)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, hyper = self.hyper))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, z_dim=self.z_dim, base=self.base, hyper = self.hyper))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes=100, base=16, z_dim=512):
    return HyperResNet(HyperBasicBlock, [2, 2, 2, 2], num_classes, base=base, z_dim=z_dim)

